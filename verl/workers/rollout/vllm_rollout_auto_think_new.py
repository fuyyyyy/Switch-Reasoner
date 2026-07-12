# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
import torch.nn.functional as F

import json
from datetime import datetime


def _to_list(x: torch.Tensor):
    return x.detach().cpu().tolist()


def _force_len_2d(x: torch.Tensor, target_len: int, pad_value: int):
    L = x.size(1)
    if L == target_len:
        return x
    if L > target_len:
        return x[:, :target_len]
    return F.pad(x, (0, target_len - L), value=pad_value)

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout_auto(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm."""
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")
        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.tokenizer = tokenizer  # ✅ ensure available
        self.processor = processor

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_sampling_params_args[key] = getattr(self.sampling_params, key)
                    setattr(self.sampling_params, key, value)
        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _has_tool_call(self, token_ids: list[int]) -> Optional[str]:
        """
        Detect <tool_call>...</tool_call> in decoded text.
        Return:
          - None: no tool call
          - str: tool call info or error string (contains "ERROR")
        """
        try:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return None

        m = _TOOL_CALL_BLOCK_RE.search(text)
        if not m:
            return None

        inner = (m.group(1) or "").strip()
        inner_l = inner.lower()

        # Very lightweight parsing (works for {"name":"require_think",...} and also plain "require_think")
        if "require_think" in inner_l:
            return (
                "I noticed you tried to call a tool. Please do not call any tools.\n"
                "Think step by step first, then provide your final answer. "
                "Put your reasoning inside <thinking>...</thinking>, and put the final answer inside <answer>...</answer>.\n"
            )
        if "think_more" in inner_l:
            return "ERROR: think_more is not allowed in Stage 0."

        return f"ERROR: unknown tool call '{inner[:80]}'"

    def truncate_keep_img_head_and_tail(
        self,
        tokens: list[int],
        max_len: int,
        img_token_id: int,
        extra_after: int = 128,
        min_tail: int = 256,
    ):
        if len(tokens) <= max_len:
            return tokens

        try:
            pos = tokens.index(img_token_id)
        except ValueError:
            return tokens[-max_len:]

        head_keep = min(max_len, pos + 1 + extra_after)
        head_keep = max(head_keep, 0)

        tail_keep = max_len - head_keep
        if tail_keep < min_tail:
            tail_keep = min_tail
            head_keep = max_len - tail_keep
            head_keep = max(0, head_keep)

        return tokens[:head_keep] + tokens[-tail_keep:]


    def _dump_rollouts_jsonl(
        self,
        dump_dir: str,
        global_step: int,
        # per-rollout aligned tensors (batch_size = trajectories count)
        input_ids: torch.Tensor,                 # (Btraj, prompt_len)
        response_ids: torch.Tensor,              # (Btraj, resp_len)
        response_mask_eos: torch.Tensor,         # (Btraj, resp_len)  (this is `response_mask` BEFORE * resp_genmask)
        resp_genmask: torch.Tensor,              # (Btraj, resp_len)  (0/1, aligned with response_ids)
        response_train_mask: torch.Tensor,       # (Btraj, resp_len)  (response_mask_eos * resp_genmask)
        # mapping
        traj_origin_index: list[int],            # len Btraj, which original sample it came from
        traj_rollout_index: list[int],           # len Btraj, which rollout k for that original sample
        max_decode_chars: int = 4000,
        only_rank0: bool = True,
    ):
        # distributed guard
        if only_rank0 and torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        os.makedirs(dump_dir, exist_ok=True)
        fname = os.path.join(
            dump_dir,
            f"rollouts_rank{self.rank}_step{global_step}.jsonl"
        )

        # decode on CPU (tokenizer is CPU anyway)
        # NOTE: decoding long strings is slow; keep it minimal.
        with open(fname, "a", encoding="utf-8") as f:
            B = response_ids.size(0)
            for j in range(B):
                prompt_ids = input_ids[j].detach().cpu().tolist()
                resp_ids = response_ids[j].detach().cpu().tolist()

                # optionally trim response by eos mask length
                eos_mask = response_mask_eos[j].detach().cpu()
                valid_len = int(eos_mask.long().sum().item())
                resp_ids_trim = resp_ids[:valid_len]

                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                response_text = self.tokenizer.decode(resp_ids_trim, skip_special_tokens=True)

                if len(prompt_text) > max_decode_chars:
                    prompt_text = prompt_text[:max_decode_chars] + "...[TRUNC]"
                if len(response_text) > max_decode_chars:
                    response_text = response_text[:max_decode_chars] + "...[TRUNC]"

                rec = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "rank": int(self.rank),
                    "global_step": int(global_step),
                    "origin_i": int(traj_origin_index[j]),
                    "rollout_k": int(traj_rollout_index[j]),
                    "prompt_ids": prompt_ids,
                    "response_ids": resp_ids_trim,  # trimmed
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "response_mask_eos": _to_list(response_mask_eos[j, :valid_len]),
                    "resp_genmask": _to_list(resp_genmask[j, :valid_len]),
                    "response_train_mask": _to_list(response_train_mask[j, :valid_len]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # import sys
            # sys.exit(0)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        bs = input_ids.size(0)

        # ---- non-tensor batch ----
        non_tensor_batch = dict(prompts.non_tensor_batch)
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        if bs != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            processed_mm = []
            vllm_inputs_round0 = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                mm = _process_multi_modal_data(
                    multi_modal_data,
                    prompts.meta_info["min_pixels"],
                    prompts.meta_info["max_pixels"],
                    prompts.meta_info["video_fps"],
                )

                processed_mm.append(mm)
                vllm_inputs_round0.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": mm})
        else:
            processed_mm = None
            vllm_inputs_round0 = [{"prompt_token_ids": list(x)} for x in batch_raw_prompt_ids]

        max_total_response_length = getattr(self.config, "max_total_response_length", self.config.response_length)
        max_round = getattr(self, "max_generation_round", 2)

        # ---- trajectory states ----
        traj_prompt_token_ids: list[list[int]] = []
        traj_prefix_lens: list[int] = []
        traj_resp_genmask: list[list[int]] = []
        traj_need_more: list[bool] = []
        traj_origin_index: list[int] = []
        traj_mult: list[int] = []  # NEW: multiplicity
        traj_mm: list[object] | None = [] if processed_mm is not None else None
        
        stop_token_ids = list(getattr(self.sampling_params, "stop_token_ids", []) or [])

        call_tools_end_ids = self.tokenizer.encode("</call_tools>", add_special_tokens=False)
        end_id = int(call_tools_end_ids[-1])
        if end_id not in stop_token_ids:
            stop_token_ids.append(end_id)

        # ---- Round 0 generate (may have n>1) ----
        with self.update_sampling_params(**prompts.meta_info, stop_token_ids=stop_token_ids):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs_round0,
                sampling_params=self.sampling_params,
                use_tqdm=self.use_tqdm,
            )

            # first-round n for prompt-side repeat
            n0 = getattr(self.sampling_params, "n", 1)
            if n0 < 1:
                n0 = 1

        traj_origin_index: list[int] = []
        traj_rollout_index: list[int] = []

        # expand to trajectories
        for i, completion in enumerate(completions):
            prefix_len = len(batch_raw_prompt_ids[i])
            base_prompt = list(batch_raw_prompt_ids[i])
            mm_i = processed_mm[i] if processed_mm is not None else None

            k = 0
            for out in completion.outputs:
                gen_ids = list(out.token_ids)
                
                need_more = self._has_tool_call(gen_ids)
                will_continue = bool(need_more) and (max_round > 1) and ("ERROR" not in str(need_more))

                full_prompt_ids = base_prompt + gen_ids
                resp_genmask = [1] * len(gen_ids)

                if will_continue:
                    tool_response_ids = self.tokenizer.encode(
                        f"<tool_response>\n{need_more}\n</tool_response>\n",
                        add_special_tokens=False,
                    )
                    full_prompt_ids.extend(tool_response_ids)
                    resp_genmask.extend([0] * len(tool_response_ids))

                traj_prompt_token_ids.append(full_prompt_ids)
                traj_prefix_lens.append(prefix_len)
                traj_resp_genmask.append(resp_genmask)
                traj_need_more.append(will_continue)

                traj_origin_index.append(i)
                traj_rollout_index.append(k)
                k += 1

                traj_mult.append(1)
                if traj_mm is not None:
                    traj_mm.append(mm_i)
                

        # ---- subsequent rounds (group & bucket by group_size; sample n=group_size) ----
        pending = [j for j, flag in enumerate(traj_need_more) if flag]
        for _round in range(1, max_round):
            if not pending:
                break

            try:
                model_max_len = self.inference_engine.llm_engine.model_config.max_model_len
            except Exception:
                model_max_len = getattr(self.config, "max_model_len", None) or (
                    self.config.prompt_length + self.config.response_length
                )

            max_prompt_len = int(
                model_max_len - int(getattr(self.sampling_params, "max_tokens", self.config.response_length))
            )
            max_prompt_len = max(16, max_prompt_len)

            pending_pos = {j: idx for idx, j in enumerate(pending)}

            groups = []
            key2gid = {}

            for j in pending:
                if len(traj_prompt_token_ids[j]) > max_prompt_len:
                    # traj_prompt_token_ids[j] = traj_prompt_token_ids[j][-max_prompt_len:]

                    has_image = (
                        traj_mm is not None
                        and traj_mm[j] is not None
                        and isinstance(traj_mm[j], dict)
                        and traj_mm[j].get("image") is not None
                        and len(traj_mm[j]["image"]) > 0
                    )

                    img_token_id = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
                    if has_image:
                        traj_prompt_token_ids[j] = self.truncate_keep_img_head_and_tail(
                            traj_prompt_token_ids[j],
                            max_prompt_len,
                            img_token_id,
                            extra_after=128,
                            min_tail=256,
                        )
                    else:
                        traj_prompt_token_ids[j] = traj_prompt_token_ids[j][-max_prompt_len:]

                mm_key = None
                if traj_mm is not None:
                    mm_key = id(traj_mm[j])

                key = (traj_origin_index[j], tuple(traj_prompt_token_ids[j]), mm_key)

                gid = key2gid.get(key, None)
                if gid is None:
                    gid = len(groups)
                    key2gid[key] = gid
                    groups.append(
                        {
                            "rep_prompt": traj_prompt_token_ids[j],
                            "rep_mm": (traj_mm[j] if traj_mm is not None else None),
                            "traj_indices": [j],
                        }
                    )
                else:
                    groups[gid]["traj_indices"].append(j)

            buckets = {}
            for gid, g in enumerate(groups):
                sz = len(g["traj_indices"])
                buckets.setdefault(sz, []).append(gid)

            next_pending_unsorted = []

            for sz in sorted(buckets.keys()):
                gid_list = buckets[sz]

                vllm_inputs_bucket = []
                for gid in gid_list:
                    g = groups[gid]
                    d = {"prompt_token_ids": g["rep_prompt"]}
                    if traj_mm is not None:
                        d["multi_modal_data"] = g["rep_mm"]
                    vllm_inputs_bucket.append(d)

                meta = dict(prompts.meta_info)
                meta["n"] = int(sz)
                with self.update_sampling_params(**meta, stop_token_ids=stop_token_ids):
                    comps: list[RequestOutput] = self.inference_engine.generate(
                        prompts=vllm_inputs_bucket,
                        sampling_params=self.sampling_params,
                        use_tqdm=self.use_tqdm,
                    )
                for gid, comp in zip(gid_list, comps):
                    g = groups[gid]
                    js = g["traj_indices"]

                    out_list = comp.outputs
                    L = min(len(js), len(out_list))

                    for k in range(L):
                        j = js[k]
                        gen_ids = list(out_list[k].token_ids)

                        need_more = self._has_tool_call(gen_ids)
                        will_continue = bool(need_more) and ("ERROR" not in str(need_more)) and (_round + 1) < max_round

                        traj_prompt_token_ids[j].extend(gen_ids)
                        traj_resp_genmask[j].extend([1] * len(gen_ids))

                        if will_continue:
                            tool_response_ids = self.tokenizer.encode(
                                f"<tool_response>\n{need_more}\n</tool_response>\n",
                                add_special_tokens=False,
                            )
                            traj_prompt_token_ids[j].extend(tool_response_ids)
                            traj_resp_genmask[j].extend([0] * len(tool_response_ids))
                            next_pending_unsorted.append(j)

            next_pending = sorted(next_pending_unsorted, key=lambda j: pending_pos.get(j, 10**18))
            pending = next_pending


        # ---- build final response_ids & masks (slice after prefix) ----
        response_ids_list: list[list[int]] = []
        resp_genmask_list: list[list[int]] = []
        for full_ids, pre_len, gm in zip(traj_prompt_token_ids, traj_prefix_lens, traj_resp_genmask):
            resp_ids = full_ids[pre_len:]
            # Sanity: gm should align with resp_ids length
            if len(gm) != len(resp_ids):
                # if mismatch, truncate to the min to avoid crashing
                L = min(len(gm), len(resp_ids))
                resp_ids = resp_ids[:L]
                gm = gm[:L]
            response_ids_list.append(resp_ids)
            resp_genmask_list.append(gm)

        extra_len = 256
        target_resp_len = max_total_response_length + extra_len

        response_ids = VF.pad_2d_list_to_length(
            response_ids_list, self.pad_token_id, max_length=target_resp_len
        ).to(input_ids.device)
        resp_genmask = VF.pad_2d_list_to_length(
            resp_genmask_list, 0, max_length=target_resp_len
        ).to(attention_mask.device)

        response_ids = _force_len_2d(response_ids, target_resp_len, self.pad_token_id)
        resp_genmask  = _force_len_2d(resp_genmask,  target_resp_len, 0)

        # ---- repeat prompt-side tensors to match traj count ----
        batch_size = len(response_ids_list)
        if n0 > 1:
            input_ids = _repeat_interleave(input_ids, n0)
            attention_mask = _repeat_interleave(attention_mask, n0)
            position_ids = _repeat_interleave(position_ids, n0)
            if batch_multi_modal_data is not None:
                batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, n0)

        # ---- concat seq / masks / position_ids ----
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)

        response_mask = VF.get_response_mask(response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype)

        final_attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        response_generation_mask = response_mask * resp_genmask.to(response_mask.dtype)

        # ---- position_ids ----
        has_mm = (traj_mm is not None) and any(mm is not None for mm in traj_mm)

        if position_ids.ndim == 3 and has_mm and (self.processor is not None):
            try:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import get_rope_index
            except Exception:
                get_rope_index = None

            if get_rope_index is None:
                delta = torch.arange(1, response_length + 1, device=position_ids.device).view(1, 1, -1)
                delta = delta.expand(batch_size, position_ids.size(1), -1)
                response_position_ids = position_ids[..., -1:] + delta
                position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            else:
                mm_for_batch = traj_mm if traj_mm is not None else [None] * batch_size

                pos_ids_list = []
                for seq_i, attn_i, mm_i in zip(sequence_ids, final_attention_mask, mm_for_batch):
                    image_grid_thw = None
                    if mm_i is not None and isinstance(mm_i, dict) and ("image" in mm_i) and (mm_i["image"] is not None):
                        img_inputs = self.processor.image_processor(mm_i["image"], return_tensors="pt")
                        image_grid_thw = img_inputs.get("image_grid_thw", None)

                    pos_i = get_rope_index(
                        self.processor,
                        input_ids=seq_i,
                        image_grid_thw=image_grid_thw,
                        attention_mask=attn_i,
                    )
                    pos_ids_list.append(pos_i)
                position_ids = torch.stack(pos_ids_list, dim=0)
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
            if position_ids.ndim == 3:
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)


        attention_mask = final_attention_mask


        dump_dir = None
        # print("dump_dir:", dump_dir)
        dump_enable = dump_dir is not None
        if dump_enable:
            global_step = int(prompts.meta_info.get("global_step", -1))
            every = int(getattr(self.config, "dump_rollouts_every", 1))
            if every <= 1 or (global_step >= 0 and global_step % every == 0):
                self._dump_rollouts_jsonl(
                    dump_dir=dump_dir,
                    global_step=global_step,
                    input_ids=input_ids,
                    response_ids=response_ids,
                    response_mask_eos=response_mask,
                    resp_genmask=resp_genmask,
                    response_train_mask=response_generation_mask,
                    traj_origin_index=traj_origin_index,
                    traj_rollout_index=traj_rollout_index,
                    max_decode_chars=int(getattr(self.config, "dump_rollouts_max_chars", 4000)),
                    only_rank0=bool(getattr(self.config, "dump_rollouts_only_rank0", True)),
                )



        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_generation_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if batch_multi_modal_data is not None:
            out_non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            out_non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=out_non_tensor_batch, meta_info=prompts.meta_info)
