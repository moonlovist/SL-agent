"""DeepSeek transport and local configuration; never archive credentials."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://api.deepseek.com/chat/completions"


def failure_message(code: str) -> str:
    return {
        'missing_api_key': '尚未配置 DeepSeek API 密钥。',
        'http_401': 'DeepSeek 密钥验证失败，请检查本地配置。',
        'http_402': 'DeepSeek 账户余额不足，恢复余额后可重新运行评审。',
        'http_429': 'DeepSeek 请求过于频繁，请稍后重试。',
        'connection_or_timeout_error': '无法连接 DeepSeek，或请求已超时。',
        'incomplete_response': 'DeepSeek 输出未完成，本次不能产生有效模型等级。',
        'invalid_or_unverifiable_response': '模型输出未通过引用、测量覆盖或格式核查，需重新评审或人工复核。',
    }.get(code, f'DeepSeek 评审未完成（{code}）。')


def local_settings() -> dict[str, str]:
    allowed = {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "LENS_AGENT_JUDGMENT"}
    settings = {}
    path = Path.cwd() / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in allowed:
                settings[key] = value.strip().strip("\"'")
    settings.update({key: os.environ[key] for key in allowed if key in os.environ})
    return settings


@dataclass(frozen=True)
class LLMConfig:
    backend: Literal["rules", "deepseek"] = "rules"
    model: str = "deepseek-v4-pro"
    timeout: float = 120.0
    max_tokens: int = 6144
    max_repairs: int = 1

    def __post_init__(self):
        if self.backend not in {"rules", "deepseek"}:
            raise ValueError("Judgment backend must be rules or deepseek")
        if not self.model.strip() or self.timeout <= 0 or self.max_tokens < 256:
            raise ValueError("Invalid DeepSeek model, timeout or token limit")
        if self.max_repairs not in {0, 1}:
            raise ValueError("At most one verification repair is allowed")

    @classmethod
    def from_environment(cls, *, backend: str | None = None, model: str | None = None):
        settings = local_settings()
        return cls(backend=backend or settings.get("LENS_AGENT_JUDGMENT", "rules"),
                   model=model or settings.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))


class LLMServiceError(RuntimeError):
    """Carries only a fixed safe error code, never the provider's response body."""


class DeepSeekClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, system_prompt: str, payload: dict) -> tuple[str, dict]:
        key = local_settings().get("DEEPSEEK_API_KEY", "")
        if not key:
            raise LLMServiceError("missing_api_key")
        body = {"model": self.config.model, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": self.config.max_tokens, "stream": False}
        request = Request(ENDPOINT, data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + key,
                                   "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("Response too large")
            result = json.loads(raw)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise LLMServiceError("incomplete_response")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty content")
            # Defensive redaction if a provider ever echoes its credential.
            content = content.replace(key, "[REDACTED]")
            metadata = {"model": str(result.get("model", self.config.model)).replace(key, "[REDACTED]"),
                        "request_id": str(result.get("id", "")).replace(key, "[REDACTED]"),
                        "usage": {k: v for k, v in (result.get("usage") or {}).items()
                                  if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                                  and isinstance(v, int) and not isinstance(v, bool)}}
            return content, metadata
        except HTTPError as exc:
            raise LLMServiceError(f"http_{exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise LLMServiceError("connection_or_timeout_error") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMServiceError("invalid_provider_response") from None
