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
"""
Format + Accuracy combined reward function for Edit-GRPO.

This reward computes:
    total_score = format_weight * format_score + accuracy_weight * accuracy_score

where:
    - format_score: checks if the response follows the expected format (e.g., has \boxed{...})
    - accuracy_score: checks if the extracted answer matches the ground truth

Usage:
    In reward config, set custom_reward_function to use this module:
        custom_reward_function:
            path: verl.utils.reward_score.format_accuracy
            name: compute_score
            reward_kwargs:
                format_weight: 1.0
                accuracy_weight: 1.0
                extract_method: boxed
"""

import re
from typing import Any, Optional


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """Extract the content of the last \boxed{...} expression."""
    idx = solution_str.rfind("\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(solution_str):
        if solution_str[i] == "{":
            num_left_braces_open += 1
        if solution_str[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return solution_str[idx + len("\boxed{"):right_brace_idx].strip()


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    if answer is None:
        return ""
    answer = answer.strip()
    answer = re.sub(r"^\$(.*)\$$", r"\1", answer)
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.rstrip(".")
    answer = answer.lower()
    return answer


def extract_answer_after_marker(solution_str: str, marker: str = "####") -> Optional[str]:
    """Extract answer after a specific marker (e.g., after '####')."""
    idx = solution_str.rfind(marker)
    if idx < 0:
        return None
    return solution_str[idx + len(marker):].strip()


def check_format(solution_str: str, require_boxed: bool = True) -> float:
    """Check if the solution follows the expected format.

    Returns 1.0 if format is correct, 0.0 otherwise.
    """
    if require_boxed:
        has_boxed = "\boxed{" in solution_str or "\boxed " in solution_str
        return 1.0 if has_boxed else 0.0
    return 1.0


def check_accuracy(
    solution_str: str,
    ground_truth: str,
    extract_method: str = "boxed",
    answer_marker: str = "####",
) -> float:
    """Check if the extracted answer matches the ground truth.

    Returns 1.0 if answer is correct, 0.0 otherwise.
    """
    if extract_method == "boxed":
        extracted = extract_boxed_answer(solution_str)
    elif extract_method == "marker":
        extracted = extract_answer_after_marker(solution_str, marker=answer_marker)
    elif extract_method == "last_number":
        numbers = re.findall(r"-?\d+\.?\d*", solution_str)
        extracted = numbers[-1] if numbers else None
    else:
        raise ValueError(f"Unknown extract_method: {extract_method}")

    if extracted is None:
        return 0.0

    extracted_norm = normalize_answer(extracted)
    ground_truth_norm = normalize_answer(str(ground_truth))

    if extracted_norm == ground_truth_norm:
        return 1.0

    try:
        extracted_num = float(extracted_norm.replace(",", ""))
        gt_num = float(ground_truth_norm.replace(",", ""))
        if abs(extracted_num - gt_num) < 1e-6:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    format_weight: float = 1.0,
    accuracy_weight: float = 1.0,
    require_boxed: bool = True,
    extract_method: str = "boxed",
    answer_marker: str = "####",
    **kwargs,
) -> dict[str, float]:
    """Compute format + accuracy combined reward for Edit-GRPO.

    Args:
        data_source: Dataset identifier (for API compatibility).
        solution_str: Model's response string.
        ground_truth: Ground truth answer.
        extra_info: Additional information (for API compatibility).
        format_weight: Weight for format score. Default: 1.0.
        accuracy_weight: Weight for accuracy score. Default: 1.0.
        require_boxed: Whether to check for \boxed{}. Default: True.
        extract_method: "boxed", "marker", or "last_number".
        answer_marker: Marker for "marker" extract method.

    Returns:
        dict with "score", "format_score", "accuracy_score".
    """
    format_score = check_format(solution_str, require_boxed=require_boxed)
    accuracy_score = check_accuracy(
        solution_str, ground_truth, extract_method=extract_method, answer_marker=answer_marker
    )

    total_score = format_weight * format_score + accuracy_weight * accuracy_score

    return {
        "score": total_score,
        "format_score": format_score,
        "accuracy_score": accuracy_score,
    }
