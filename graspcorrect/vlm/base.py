from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from graspcorrect.types import GraspCandidate


GRASP_GUIDED_PROMPT = """You are a robot equipped with a parallel-jaw gripper, performing the task '{task_desc}'.
Analyze the provided pre-grasp pose of an object and specify precise contact positions
for each robot gripper to achieve a stable grasp. Describe the contact position as
much detail as possible using numerical expressions. Avoid using exact coordinates.
Respond in the format: 'Left: [1 sentence starting with ''Position the left
gripper'']. Right: [1 sentence starting with ''Position the right gripper'']'. Let's
think step by step."""


ITERATIVE_VQA_PROMPT = """INSTRUCTIONS: You are tasked to locate an object, region, or point in space in the
given annotated image according to a description. The image is annotated with numbered
circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what
the description is describing in the image. You are a five-time world champion in this
game. Give a one sentence analysis of why you chose those points. Provide your answer
at the end in a valid JSON of this format: "points": [].
DESCRIPTION: {description}
IMAGE: {image_hint}"""


class VLMClient(Protocol):
    def describe_grasp(self, image: np.ndarray, task_desc: str) -> str:
        ...

    def choose_points(self, image: np.ndarray, description: str, top_n: int) -> list[int]:
        ...


@dataclass
class HeuristicVLMClient:
    """Offline substitute used for tests and smoke runs.

    The detector still performs object-aware contour sampling. This client only
    avoids remote VLM calls by returning generic contact descriptions and the
    first labels requested by the detector.
    """

    def describe_grasp(self, image: np.ndarray, task_desc: str) -> str:
        return (
            "Left: Position the left gripper on one exposed lateral edge of the target object. "
            "Right: Position the right gripper on the opposite lateral edge at a similar height."
        )

    def choose_points(self, image: np.ndarray, description: str, top_n: int) -> list[int]:
        return list(range(1, top_n + 1))


def parse_points_json(text: str) -> list[int]:
    """Extract the PIVOT-style {"points": [...]} answer from VLM text."""

    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    brace_match = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    candidates.extend(brace_match)
    candidates.append(text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        points = parsed.get("points") if isinstance(parsed, dict) else parsed
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
            out = []
            for point in points:
                try:
                    out.append(int(point))
                except (TypeError, ValueError):
                    continue
            if out:
                return out

    loose = re.findall(r"\b\d+\b", text)
    if loose:
        return [int(x) for x in loose]
    raise ValueError(f"Could not parse VLM point JSON from: {text!r}")


def candidate_numbers(candidates: Sequence[GraspCandidate]) -> str:
    return ", ".join(str(candidate.label) for candidate in candidates)
