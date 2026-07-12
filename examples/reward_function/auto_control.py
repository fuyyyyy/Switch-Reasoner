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

import re
from typing import Any, Optional

from mathruler.grader import grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"

MATH_DATASET_IDS = ("geometry3k", "mathvista")
IOU_DATASET_IDS = ("refadv",)
IOU_THRESHOLD = 0.5
DOCVQA_DATASET_IDS = ("docvqa",)
CHARTQA_DATASET_IDS = ("chartqa",)

SIMPLE_ANSWER_RE = re.compile(r"\A\s*<answer>.*?</answer>\s*\Z", re.DOTALL | re.IGNORECASE)

SECOND_FMT_RE = re.compile(
    r"\A\s*"
    r"<tool_call>\s*.*?\s*</tool_call>\s*"
    r"<thinking>\s*.*?\s*</thinking>\s*"
    r"<answer>\s*.*?\s*</answer>\s*"
    r"\Z",
    re.DOTALL | re.IGNORECASE
)

TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)


def format_reward(response: str) -> float:
    return float(bool(SIMPLE_ANSWER_RE.fullmatch(response) or SECOND_FMT_RE.fullmatch(response)))


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _dataset_id(sample_id: Any) -> str:
    return _normalize_text(sample_id)


def _direct_match(answer: Any, ground_truth: Any) -> bool:
    return any(_normalize_text(answer) == _normalize_text(gt) for gt in _ground_truths(ground_truth))


def _ground_truths(ground_truth: Any) -> list[Any]:
    if isinstance(ground_truth, (list, tuple)):
        return list(ground_truth)
    if not isinstance(ground_truth, (str, bytes)) and hasattr(ground_truth, "tolist"):
        value = ground_truth.tolist()
        return value if isinstance(value, list) else [value]
    return [ground_truth]


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _docvqa_anls(answer: Any, ground_truth: Any) -> float:
    prediction = _normalize_text(answer)
    scores = []
    for gt in _ground_truths(ground_truth):
        target = _normalize_text(gt)
        max_length = max(len(prediction), len(target))
        similarity = 1.0 if max_length == 0 else 1.0 - _edit_distance(prediction, target) / max_length
        scores.append(similarity if similarity >= 0.5 else 0.0)
    return max(scores, default=0.0)


def _to_float(value: Any) -> Optional[float]:
    text = str(value).strip()
    try:
        return float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError:
        return None


def _chartqa_relaxed_accuracy(answer: Any, ground_truth: Any, max_relative_change: float = 0.05) -> float:
    prediction = str(answer).strip()
    prediction_float = _to_float(prediction)
    for gt in _ground_truths(ground_truth):
        target = str(gt).strip()
        target_float = _to_float(target)
        if prediction_float is not None and target_float:
            if abs(prediction_float - target_float) / abs(target_float) <= max_relative_change:
                return 1.0
        elif prediction.lower() == target.lower():
            return 1.0
    return 0.0


def _extract_bbox(value: Any) -> Optional[list[float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(nums) < 4:
        return None
    return [float(num) for num in nums[:4]]


def _bbox_iou(prediction: Any, ground_truth: Any) -> float:
    pred_box = _extract_bbox(prediction)
    gt_box = _extract_bbox(ground_truth)
    if pred_box is None or gt_box is None:
        return 0.0

    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    px1, px2 = sorted((px1, px2))
    py1, py2 = sorted((py1, py2))
    gx1, gx2 = sorted((gx1, gx2))
    gy1, gy2 = sorted((gy1, gy2))

    inter_w = max(0.0, min(px2, gx2) - max(px1, gx1))
    inter_h = max(0.0, min(py2, gy2) - max(py1, gy1))
    inter_area = inter_w * inter_h
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def accuracy_reward(response: str, ground_truth: str, sample_id: Any = None) -> float:
    _ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    m = _ANSWER_RE.search(response)
    answer = m.group(1) if m else response
    dataset_id = _dataset_id(sample_id)
    if any(name in dataset_id for name in DOCVQA_DATASET_IDS):
        return _docvqa_anls(answer, ground_truth)
    if any(name in dataset_id for name in CHARTQA_DATASET_IDS):
        return _chartqa_relaxed_accuracy(answer, ground_truth)
    if any(name in dataset_id for name in IOU_DATASET_IDS):
        return 1.0 if _bbox_iou(answer, ground_truth) >= IOU_THRESHOLD else 0.0
    if sample_id is None or any(name in dataset_id for name in MATH_DATASET_IDS):
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    return 1.0 if _direct_match(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.2,
    *,
    penalty_mul: float = 0.9,
    eps: float = 0.2,
    b_clip: float = 5.0,
    apply_only_if_format_ok: bool = True,
    apply_only_if_accuracy_ok: bool = False,
    must_think_weight: float = 0.03,
    safe_direct_weight: float = 0.03,
    must_think_margin: float = 0.5,
    safe_direct_min_success: float = 0.75,
    safe_direct_max_gain: float = -0.25,
) -> list[dict[str, float]]:
    """
     reward:
        overall = (1-format_weight)*accuracy + format_weight*format
    B-only :
        if B>eps:    think (is_thinking==1) => overall *= 0.9
        if B<-eps:   nothink (is_thinking==0) => overall *= 0.9

    Per-sample route reward:
        - must-think: think  direct 
        - safe-direct: direct ， think 
        - uncertain: 

    B ， per-sample 。
    """
    scores = []
    fw = float(format_weight)

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        response = re.sub(
            r"<tool_response>\s*.*?\s*</tool_response>\s*user\s*.*?\s*assistant",
            "",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        format_score = float(format_reward(response))
        accuracy_score = float(accuracy_reward(response, reward_input["ground_truth"], reward_input.get("id")))
        overall_score = (1.0 - fw) * accuracy_score + fw * format_score

        thinking_tag = 0.0 if SIMPLE_ANSWER_RE.fullmatch(response) else 1.0

        B = float(reward_input.get("B") or 0.0)
        if B > b_clip:
            B = b_clip
        elif B < -b_clip:
            B = -b_clip

        can_apply = True
        if apply_only_if_format_ok and not (format_score > 0.5):
            can_apply = False
        if apply_only_if_accuracy_ok and not (accuracy_score > 0.5):
            can_apply = False

        # B only prevents global collapse. Per-sample route supervision remains
        # active whenever the counterfactual route can be classified.
        did_think = thinking_tag > 0.5
        route_delta = 0.0
        route_reward = 0.0
        route_must_think = False
        route_safe_direct = False
        route_uncertain = False
        route_case_active = False
        route_choice_correct = False
        route_false_nothink = False
        route_false_think = False
        route_must_think_chose_think = False
        route_safe_direct_chose_direct = False

        if abs(B) > eps:
            if can_apply:
                if B > eps and did_think:
                    overall_score *= float(penalty_mul)

                elif B < -eps and (not did_think):
                    overall_score *= float(penalty_mul)

        direct_success = reward_input.get("route_direct_success")
        think_success = reward_input.get("route_think_success")
        if direct_success is not None and think_success is not None:
            route_case_active = True
            direct_success = float(direct_success)
            think_success = float(think_success)
            route_delta = think_success - direct_success

            route_must_think = route_delta >= float(must_think_margin)
            route_safe_direct = (
                direct_success >= float(safe_direct_min_success)
                and route_delta <= float(safe_direct_max_gain)
            )
            route_uncertain = not (route_must_think or route_safe_direct)

            route_false_nothink = route_must_think and (not did_think)
            route_false_think = route_safe_direct and did_think
            route_must_think_chose_think = route_must_think and did_think
            route_safe_direct_chose_direct = route_safe_direct and (not did_think)
            route_choice_correct = route_must_think_chose_think or route_safe_direct_chose_direct

            # Route supervision is independent of answer correctness. Keep the
            # format guard because malformed output does not reveal a reliable
            # think/direct decision.
            if format_score > 0.5:
                if route_must_think:
                    route_reward = float(must_think_weight) if did_think else -float(must_think_weight)
                elif route_safe_direct:
                    route_reward = float(safe_direct_weight) if not did_think else -float(safe_direct_weight)

            overall_score += route_reward
        scores.append(
            {
                "overall": float(overall_score),
                "format": float(format_score),
                "accuracy": float(accuracy_score),
                "is_thinking": float(thinking_tag),
                "route_reward": float(route_reward),
                "route_delta": float(route_delta),
                "route_case_active": float(route_case_active),
                "route_must_think": float(route_must_think),
                "route_safe_direct": float(route_safe_direct),
                "route_uncertain": float(route_uncertain),
                "route_choice_correct": float(route_choice_correct),
                "route_false_nothink": float(route_false_nothink),
                "route_false_think": float(route_false_think),
                "route_must_think_chose_think": float(route_must_think_chose_think),
                "route_safe_direct_chose_direct": float(route_safe_direct_chose_direct),
                "route_false_nothink_error": float(route_false_nothink and accuracy_score <= 0.5),
                "route_false_think_error": float(route_false_think and accuracy_score <= 0.5),
                "route_direct_success": float(reward_input.get("route_direct_success", 0.0)),
                "route_think_success": float(reward_input.get("route_think_success", 0.0)),
            }
        )

    return scores
