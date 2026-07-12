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

import math
import os
import json
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        assist_key: str = "assist",
        id_key: str = "id",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        user_direct_prompt: Optional[str] = None,
        user_think_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        filter_overlong_log_path: Optional[str] = None,
        data_list: list = None,
        is_path: bool = True,
        mode: str = "naive",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.assist_key = assist_key
        self.id_key = id_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.filter_overlong_log_path = filter_overlong_log_path
        self.is_path = is_path
        self.mode = mode

        if is_path:
            if "@" in data_path:
                data_path, data_split = data_path.split("@")
            else:
                data_split = "train"

            if os.path.isdir(data_path):
                # when we use dataset builder, we should always refer to the train split
                file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
                self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
            elif os.path.isfile(data_path):
                file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
                self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
            else:
                # load remote dataset from huggingface hub
                self.dataset = load_dataset(data_path, split=data_split)

            self.system_prompt = None
            if system_prompt:
                with open(system_prompt, encoding="utf-8") as f:
                    self.system_prompt = f.read()

            self.user_prompt = None
            if user_prompt:
                with open(user_prompt, encoding="utf-8") as f:
                    self.user_prompt = f.read()

            self.user_direct_prompt = None
            if user_direct_prompt:
                with open(user_direct_prompt, encoding="utf-8") as f:
                    self.user_direct_prompt = f.read()

            self.user_think_prompt = None
            if user_think_prompt:
                with open(user_think_prompt, encoding="utf-8") as f:
                    self.user_think_prompt = f.read()

            self.tools = [
                {
                    "type": "function",
                    "function": {
                    "name": "require_think",
                    "description": "Request a thinking upgrade.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
            ]

            if self.image_key in self.dataset[0]:
               self.raw_images = [ex[self.image_key] for ex in self.dataset]

            if filter_overlong_prompts:
                self.dataset = self.dataset.map(
                    self._measure_prompt_length,
                    desc="Measuring prompt lengths",
                    num_proc=filter_overlong_prompts_workers,
                    with_indices=True,
                )
                self._write_overlong_prompts_log()
                self.dataset = self.dataset.filter(
                    self._filter_overlong_prompts,
                    desc="Filtering overlong prompts",
                    num_proc=filter_overlong_prompts_workers,
                )
                self.dataset = self.dataset.remove_columns(["__prompt_length", "__dataset_index"])


        else:
            self.dataset = data_list
            self.system_prompt = None
            if system_prompt:
                with open(system_prompt, encoding="utf-8") as f:
                    self.system_prompt = f.read()
            if self.image_key in self.dataset[0]:
               self.raw_images = [ex[self.image_key] for ex in self.dataset]


    def _build_messages(
        self, example: dict[str, Any], use_direct: bool = False, use_think: bool = False
    ) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]

        if self.user_prompt and not use_direct and not use_think:
            t = Template(self.user_prompt.strip())
            prompt_str = t.render(content=prompt_str)

        if use_direct and self.user_direct_prompt:
            t = Template(self.user_direct_prompt.strip())
            prompt_str = t.render(content=prompt_str)

        if use_think and self.user_think_prompt:
            t = Template(self.user_think_prompt.strip())
            prompt_str = t.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            if self.mode == "naive":
                return [{"role": "user", "content": content_list}]
            elif self.mode == "auto":
                if use_direct or use_think:
                    return [{"role": "user", "content": content_list}]
                else:
                    return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content_list}]

        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _measure_prompt_length(self, example: dict[str, Any], index: int) -> dict[str, int]:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, tools = self.tools, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            prompt_length = model_inputs["input_ids"].size(-1)
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            prompt_length = model_inputs["input_ids"].size(-1)
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            prompt_length = len(input_ids)

        return {"__prompt_length": int(prompt_length), "__dataset_index": index}


    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, tools = self.tools, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def _write_overlong_prompts_log(self) -> None:
        if not self.filter_overlong_log_path:
            return

        overlong_dataset = self.dataset.filter(
            lambda example: example["__prompt_length"] > self.max_prompt_length,
            desc="Collecting overlong prompts",
        )
        os.makedirs(os.path.dirname(self.filter_overlong_log_path), exist_ok=True)
        with open(self.filter_overlong_log_path, "w", encoding="utf-8") as f:
            for example in overlong_dataset:
                dataset_index = int(example["__dataset_index"])
                record = {
                    "id": example.get(self.id_key, example.get("id", example.get("sample_ids", dataset_index))),
                    "dataset_index": dataset_index,
                    "prompt_length": int(example["__prompt_length"]),
                    "max_prompt_length": self.max_prompt_length,
                    "overflow_tokens": int(example["__prompt_length"]) - self.max_prompt_length,
                    "prompt": example.get(self.prompt_key),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Logged {len(overlong_dataset)} overlong prompts to {self.filter_overlong_log_path}.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        messages_direct = self._build_messages(example, use_direct=True)
        messages_think = self._build_messages(example, use_think=True)
        prompt_direct = None
        prompt_think = None
        example["sample_ids"] = example.get(self.id_key, example.get("id", example.get("sample_ids", index)))
        example["raw_prompts"] = example[self.prompt_key]
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            if self.mode == "naive":
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_direct = None
            elif self.mode == "auto":
                prompt = self.processor.apply_chat_template(messages, tools=self.tools, add_generation_prompt=True, tokenize=False)
                prompt_direct = self.processor.apply_chat_template(messages_direct, add_generation_prompt=True, tokenize=False)
                prompt_think = self.processor.apply_chat_template(messages_think, add_generation_prompt=True, tokenize=False)

            if not self.is_path:
                prompt += str(example[self.assist_key])
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}

            example["raw_images"] = self.raw_images[index]

        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            if self.mode == "auto":
                prompt_direct = self.tokenizer.apply_chat_template(messages_direct, add_generation_prompt=True, tokenize=False)
                prompt_think = self.tokenizer.apply_chat_template(messages_think, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if prompt_direct is not None:
            raw_prompt_ids_direct = self.tokenizer.encode(prompt_direct, add_special_tokens=False)
        if prompt_think is not None:
            raw_prompt_ids_think = self.tokenizer.encode(prompt_think, add_special_tokens=False)

        def _truncate(ids, name):
            if len(ids) <= self.max_prompt_length:
                return ids
            if self.truncation == "left":
                return ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                return ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"{name} length {len(ids)} is longer than {self.max_prompt_length}.")
            return ids[: self.max_prompt_length]

        raw_prompt_ids = _truncate(raw_prompt_ids, "Prompt")
        if prompt_direct is not None:
            raw_prompt_ids_direct = _truncate(raw_prompt_ids_direct, "Direct prompt")
        if prompt_think is not None:
            raw_prompt_ids_think = _truncate(raw_prompt_ids_think, "Think prompt")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        if prompt_direct is not None:
            example["raw_prompt_ids_direct"] = raw_prompt_ids_direct
        if prompt_think is not None:
            example["raw_prompt_ids_think"] = raw_prompt_ids_think
        example["ground_truth"] = example.pop(self.answer_key)
        return example
