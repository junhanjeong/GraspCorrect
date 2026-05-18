from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

from graspcorrect.config import VLMConfig, load_dotenv
from graspcorrect.utils.image import image_to_data_url


GRASP_GUIDED_PROMPT = """You are a robot equipped with a parallel-jaw gripper, performing the task '{task_desc}'.
Analyze the provided pre-grasp pose of an object and specify precise contact positions
for each robot gripper to achieve a stable grasp. Describe the contact position as
much detail as possible using numerical expressions. Avoid using exact coordinates.
Respond in the format: 'Left: [1 sentence starting with "Position the left
gripper"]. Right: [1 sentence starting with "Position the right gripper"]'. Let's
think step by step."""


ITERATIVE_VQA_PROMPT = """INSTRUCTIONS: You are tasked to locate an object, region, or point in space in the
given annotated image according to a description. The image is annotated with numbered
circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what
the description is describing in the image. You are a five-time world champion in this
game. Give a one sentence analysis of why you chose those points. Provide your answer
at the end in a valid JSON of this format: {{"points": []}}.
DESCRIPTION: {description}"""


class VLMClient:
    def describe_grasp(self, image: np.ndarray, task_desc: str) -> str:
        raise NotImplementedError

    def choose_points(self, annotated_image: np.ndarray, description: str, top_n: int) -> List[int]:
        raise NotImplementedError


@dataclass
class OpenAIGPT54MiniVLM(VLMClient):
    """GPT-5.4 mini VQA client using OPENAI_API_KEY from .env."""

    config: VLMConfig = field(default_factory=VLMConfig)

    def __post_init__(self) -> None:
        load_dotenv(".env")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise ImportError("Install the OpenAI SDK in this environment.") from exc
        self.client = OpenAI()

    def describe_grasp(self, image: np.ndarray, task_desc: str) -> str:
        return self._ask_image(GRASP_GUIDED_PROMPT.format(task_desc=task_desc), image)

    def choose_points(self, annotated_image: np.ndarray, description: str, top_n: int) -> List[int]:
        text = self._ask_image(ITERATIVE_VQA_PROMPT.format(top_n=top_n, description=description), annotated_image)
        return parse_points_json(text)[:top_n]

    def _ask_image(self, prompt: str, image: np.ndarray) -> str:
        response = self.client.responses.create(
            model=self.config.model,
            temperature=self.config.temperature,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": self.config.detail},
                    ],
                }
            ],
        )
        text = getattr(response, "output_text", None)
        if text:
            return str(text)
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(str(value))
        if not chunks:
            raise RuntimeError("OpenAI response did not include text.")
        return "\n".join(chunks)


def parse_points_json(text: str) -> List[int]:
    matches = re.findall(r"\{[^{}]*\"points\"[^{}]*\}", text, flags=re.DOTALL)
    for raw in reversed(matches):
        try:
            obj = json.loads(raw)
            pts = obj.get("points", [])
            return [int(x) for x in pts]
        except Exception:
            continue
    match = re.search(r"points\s*[:=]\s*\[([^\]]*)\]", text, flags=re.IGNORECASE)
    if match:
        return [int(x) for x in re.findall(r"\d+", match.group(1))]
    return [int(x) for x in re.findall(r"\b\d+\b", text)]


# Backward-compatible alias for older imports.
OpenAIChatGPT4oVLM = OpenAIGPT54MiniVLM
