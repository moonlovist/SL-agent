import os
import re
import json
import asyncio
import logging
from pathlib import Path

from FORMA.agents.common.message_utils import create_message
from FORMA.core.llm import _detect_vendor, _build_thinking_extra_body

# ── 连接错误关键字，断网/重联时会触发重试 ──
_CONNECTION_ERROR_KEYWORDS = (
    "connection reset",
    "connection refused",
    "remotedisconnected",
    "clientconnectorerror",
    "apiconnectionerror",
    "connect error",
    "network",
    "broken pipe",
    "eof occurred",
    "ssl",
)

# ── 超时错误关键字 ──
_TIMEOUT_KEYWORDS = (
    "connectionerror",
    "connecttimeout",
    "timed out",
    "timeout",
    "readtimeout",
    "read timeout",
)

# ── API 限速/配额错误关键字 ──
_RATE_LIMIT_KEYWORDS = (
    "rate limit",
    "insufficient_quota",
    "invalid_parameter_error",
    "function.arguments",
)


def _is_connection_error(error_msg: str) -> bool:
    """判断是否为连接层错误（断网、连接拒绝等）。"""
    return any(kw in error_msg for kw in _CONNECTION_ERROR_KEYWORDS)


def _is_timeout_error(error_msg: str) -> bool:
    """判断是否为超时错误。"""
    return any(kw in error_msg for kw in _TIMEOUT_KEYWORDS)


def _is_retryable_error(error_msg: str) -> bool:
    """判断是否为可重试错误（连接错误 + 超时 + 限速错误）。"""
    return (
        _is_connection_error(error_msg)
        or _is_timeout_error(error_msg)
        or any(kw in error_msg for kw in _RATE_LIMIT_KEYWORDS)
    )


class BaseAgent:

    agent_name: str = "BaseAgent"

    # Override in subclasses to point to their skill directory.
    _SKILL_DIR: Path | None = None

    def __init__(self, runtime):
        self.runtime = runtime

        self._text_model = runtime.get_model("llm")

        # ── VLM 惰性加载（PNG 通道已注释，VLM 可选）───────────────
        self._vis_model = None
        vlm_cfg = runtime.configs.model.vlm
        if vlm_cfg.get('api_key') and vlm_cfg.get('model') and vlm_cfg.get('base_url'):
            self._vis_model = runtime.get_model("vlm")

        # thinking mode config ('none' = parameter not supported by this provider)
        self._llm_thinking = runtime.configs.model.llm.get('thinking', 'disabled')
        self._vlm_thinking = runtime.configs.model.vlm.get('thinking', 'none')

        # auto-detect vendor from base_url for correct thinking extra_body format
        self._llm_vendor = _detect_vendor(runtime.configs.model.llm.get('base_url', ''))
        self._vlm_vendor = _detect_vendor(runtime.configs.model.vlm.get('base_url', ''))

    # ------------------------------------------------------------------
    # Skill helpers
    # ------------------------------------------------------------------

    def _skill_path(self, name: str) -> Path:
        """Resolve a skill file path relative to ``_SKILL_DIR``.

        If *name* has no extension, ``.md`` is appended automatically.
        Subclasses must set ``_SKILL_DIR``.
        """
        if self._SKILL_DIR is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _SKILL_DIR to use _load_skill()"
            )
        return self._SKILL_DIR / (name if name.endswith(".md") else f"{name}.md")

    def _load_skill(self, name: str) -> str:
        """Load a skill markdown file from ``_SKILL_DIR``."""
        return self._skill_path(name).read_text(encoding="utf-8")

    # --------------------------
    # Thinking helper
    # --------------------------

    def _apply_thinking(self, model, thinking_cfg: str, vendor: str, want_tools: bool):
        """Return a model with appropriate thinking extra_body bound.

        Base model already has the configured thinking state baked in.
        We only need to override when want_tools=True but thinking=enabled:
        force disabled to avoid reasoning_content passback issue in LangGraph.
        """
        if thinking_cfg == 'none' or vendor == 'unknown':
            return model
        if want_tools and thinking_cfg == 'enabled':
            # Override to disabled — multi-turn tool calls can't pass back reasoning_content
            extra_body = _build_thinking_extra_body('disabled', vendor)
            return model.bind(extra_body=extra_body)
        return model

    # --------------------------
    # Call
    # --------------------------

    async def call_llm_with_context(
        self,
        system_prompt,
        user_prompt,
        image_path=None,
        parse_json=True,
        description="LLM output",
    ):

        max_retries = self.runtime.configs.max_tries or 3
        retry_delay = self.runtime.configs.retry_delay or 180

        for attempt in range(max_retries + 1):

            try:
                messages = create_message(
                    system_prompt,
                    user_prompt,
                    image_path
                )

                if image_path:
                    if self._vis_model is None:
                        raise RuntimeError(
                            "VLM 未配置（VLM_API_KEY / VLM_MODEL / VLM_BASE_URL 为空）。"
                            "PNG 通道已注释，当前仅支持 FITS 输入。"
                        )
                    vlm = self._apply_thinking(self._vis_model, self._vlm_thinking, self._vlm_vendor, want_tools=False)
                    response = await vlm.ainvoke(messages)
                    raw_content = response.content
                else:
                    llm = self._apply_thinking(self._text_model, self._llm_thinking, self._llm_vendor, want_tools=False)
                    response = await llm.ainvoke(messages)
                    raw_content = response.content

                # ---------------- JSON parse ----------------

                if parse_json:
                    cleaned = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        raw_content.strip(),
                        flags=re.IGNORECASE,
                    ).strip()

                    try:
                        return json.loads(cleaned)
                    except Exception:
                        pass
                    # Fallback: LLM may return Python-style literals (single quotes)
                    try:
                        import ast
                        parsed = ast.literal_eval(cleaned)
                        if isinstance(parsed, (dict, list)):
                            return parsed
                    except Exception:
                        pass
                    return cleaned

                return raw_content

            except Exception as e:

                error_msg = str(e).lower()

                if attempt < max_retries and _is_retryable_error(error_msg):
                    if _is_connection_error(error_msg):
                        # connection error: retry after delay
                        logging.warning(
                            f"🌐 {description} connection error, retrying..."
                            f" (attempt {attempt + 1}/{max_retries}): {str(e)}"
                        )
                    elif _is_timeout_error(error_msg):
                        # 超时错误：仅等待后重试
                        logging.warning(
                            f"⏱️ {description}遇到超时，{retry_delay}秒后重试..."
                            f" (尝试 {attempt + 1}/{max_retries})"
                        )
                    else:
                        # 限速/配额错误：等待后重试，不需要重置连接
                        logging.warning(
                            f"⏳ {description}遇到限制，{retry_delay}秒后重试..."
                            f" (尝试 {attempt + 1}/{max_retries})"
                        )
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"❌ {description}失败: {str(e)}")
                    raise
