from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from graspcorrect.baselines.base import PythonClassPolicyAdapter, SubprocessPolicyAdapter


@dataclass
class DiffuserActorAdapterFactory:
    """Factory for wiring the official 3D Diffuser Actor implementation.

    The official repository exposes different entry points for RLBench and
    CALVIN experiments. Use `python_import` when you have a local inference
    class; otherwise use `command` for a JSON bridge script inside that repo.
    """

    root: Path
    python_import: Optional[str] = None
    command: Optional[list[str]] = None

    def create(self):
        if self.python_import:
            return PythonClassPolicyAdapter(self.python_import, kwargs={"root": str(self.root)})
        if self.command:
            return SubprocessPolicyAdapter(self.command, cwd=self.root)
        raise ValueError(
            "Provide either python_import='module:ClassName' or command=[...] for the local "
            "3D Diffuser Actor checkout."
        )
