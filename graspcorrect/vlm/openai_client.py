from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graspcorrect.utils.image import image_to_data_url
from graspcorrect.vlm.base import GRASP_GUIDED_PROMPT, ITERATIVE_VQA_PROMPT, parse_points_json


@dataclass
class OpenAIResponsesVLMClient:
    """OpenAI Responses API client for the VQA calls used by GraspCorrect."""

    model: str = "gpt-4o"
    temperature: float = 0.0
    detail: str = "high"

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the OpenAI SDK with `pip install -e .[vlm]`.") from exc
        self.client = OpenAI()

    def describe_grasp(self, image: np.ndarray, task_desc: str) -> str:
        prompt = GRASP_GUIDED_PROMPT.format(task_desc=task_desc)
        return self._ask_image(prompt, image)

    def choose_points(self, image: np.ndarray, description: str, top_n: int) -> list[int]:
        prompt = ITERATIVE_VQA_PROMPT.format(top_n=top_n, description=description, image_hint="<attached image>")
        text = self._ask_image(prompt, image)
        return parse_points_json(text)

    def _ask_image(self, prompt: str, image: np.ndarray) -> str:
        response = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(image),
                            "detail": self.detail,
                        },
                    ],
                }
            ],
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        if not chunks:
            raise RuntimeError("OpenAI response did not contain text output.")
        return "\n".join(chunks)
