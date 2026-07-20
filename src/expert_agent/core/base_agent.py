from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path

from .state import ExpertState


class BaseAgent(ABC):
    """
    所有专家推理 Agent 的基础类。

    每个 Agent:
    1. 接收 ExpertState
    2. 读取已有信息
    3. 加载对应专家技能
    4. 执行推理
    5. 修改状态
    6. 返回状态
    """

    def __init__(self, name: str):

        self.name = name

        self.skill_dir = (
            Path(__file__).parent.parent / "skills"
        )


    def log(self, message: str):
        print(
            f"[{self.name}] {message}"
        )


    def load_skill(self, skill_name: str) -> str:
        """
        读取专家规则文件。

        Example:
        self.load_skill(
            "observation_skill"
        )
        """

        path = (
            self.skill_dir /
            f"{skill_name}.md"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Skill not found: {path}"
            )

        return path.read_text(
            encoding="utf-8"
        )


    def record_reasoning(
        self,
        state: ExpertState,
        action: str,
        content: Dict[str, Any]
    ):

        state.reasoning_trace.append(
            {
                "agent": self.name,
                "action": action,
                "content": content
            }
        )


    def build_prompt(
        self,
        skill: str,
        task: str
    ) -> str:
        """
        构造专家任务prompt。
        """

        return f"""
You are {self.name}.

Follow this expert reasoning skill:

----------------
{skill}
----------------

Your task:

{task}
"""


    def call_llm(
        self,
        prompt: str
    ):
        """
        LLM接口预留。

        后续接入:
        GPT / Claude / Qwen / local model
        """

        raise NotImplementedError(
            "LLM backend not connected yet."
        )


    @abstractmethod
    def run(
        self,
        state: ExpertState
    ) -> ExpertState:
        pass