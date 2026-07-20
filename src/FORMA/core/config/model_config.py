import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

from FORMA.core.config._utils import getenv_float


# ------------------------
# LLM / VLM 配置
# ------------------------

class ModelConfig(BaseModel):
    llm: Dict[str, Any]
    vlm: Dict[str, Any]

    @classmethod
    def from_env(cls) -> "ModelConfig":
        def parse_max_tokens(v: Optional[str]) -> Optional[int]:
            if v and v.strip():
                try:
                    return int(v.strip())
                except ValueError:
                    print(f"⚠️ MAX_TOKENS 格式错误: {v}，使用 None")
            return None

        llm={
            "api_key": os.getenv("LLM_API_KEY"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "model": os.getenv("LLM_MODEL"),
            "temperature": getenv_float("LLM_TEMPERATURE", 0.1),
            "max_tokens": parse_max_tokens(os.getenv("LLM_MAX_TOKENS")),
            "thinking": os.getenv("LLM_THINKING", "disabled"),
        }
        vlm={
            "api_key": os.getenv("VLM_API_KEY"),
            "base_url": os.getenv("VLM_BASE_URL"),
            "model": os.getenv("VLM_MODEL"),
            "temperature": getenv_float("VLM_TEMPERATURE", 0.1),
            "max_tokens": parse_max_tokens(os.getenv("VLM_MAX_TOKENS")),
            "thinking": os.getenv("VLM_THINKING", "disabled"),
        }

        # ── LLM 配置校验（必填）──────────────────────────────────
        if not all([llm['api_key'], llm['model'], llm["base_url"]]):
            raise ValueError("LLM 配置不完整（LLM_API_KEY / LLM_MODEL / LLM_BASE_URL 必填）")

        # ── VLM 配置（可选，PNG 通道已注释暂不需要）────────────────
        # 如果 VLM 未配置，vlm 字典保留空值，base_agent 会在首次调用时跳过

        return cls(llm=llm, vlm=vlm)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()