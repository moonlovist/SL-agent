from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .state import ExpertState


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.skill_dir = Path(__file__).parent.parent / "skills"

    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}")

    def load_skill(self, skill_name: str) -> str:
        path = self.skill_dir / f"{skill_name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Skill not found: {path}")
        return path.read_text(encoding="utf-8")

    def record_reasoning(
        self, state: ExpertState, action: str, content: dict[str, Any]
    ) -> None:
        state.reasoning_trace.append(
            {"agent": self.name, "action": action, "content": content}
        )

    @abstractmethod
    def run(self, state: ExpertState) -> ExpertState:
        raise NotImplementedError
