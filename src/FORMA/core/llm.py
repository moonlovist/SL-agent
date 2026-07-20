import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI


def _detect_vendor(base_url: str) -> str:
    """根据 base_url 判断模型厂商。

    Returns:
        'deepseek' | 'qwen' | 'unknown'
    """
    if not base_url:
        return 'unknown'
    url = base_url.lower()
    if 'deepseek' in url:
        return 'deepseek'
    if 'aliyuncs' in url or 'dashscope' in url or 'qwen' in url:
        return 'qwen'
    return 'unknown'


def _build_thinking_extra_body(mode: str, vendor: str):
    """根据厂商和模式构建 thinking 相关的 extra_body。

    Args:
        mode:   'enabled' | 'disabled'
        vendor: 'deepseek' | 'qwen' | 'unknown'

    Returns:
        dict or None
    """
    if vendor == 'deepseek':
        # DeepSeek API: {"thinking": {"type": "enabled/disabled"}}
        return {'thinking': {'type': mode}}
    if vendor == 'qwen':
        # Qwen3 API: {"enable_thinking": True/False}
        return {'enable_thinking': mode == 'enabled'}
    # 未知厂商：不传，避免参数不兼容
    return None


def _create_chat_openai(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    streaming: bool = False,
    extra_body: dict | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance.

    For DeepSeek, patches ``_get_request_payload`` to keep ``max_tokens`` in the
    API request body.  langchain-openai >= 1.x unconditionally renames
    ``max_tokens`` → ``max_completion_tokens`` (for OpenAI API compat), but
    DeepSeek only recognises ``max_tokens`` and ignores ``max_completion_tokens``,
    silently falling back to its default (8192).
    """
    vendor = _detect_vendor(base_url)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        extra_body=extra_body,
    )

    if vendor == "deepseek" and max_tokens is not None:
        _orig = llm._get_request_payload

        def _patched(self, input_, *, stop=None, **kwargs):
            payload = _orig(input_, stop=stop, **kwargs)
            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
            return payload

        # Bind as instance method so `self` is automatically passed
        llm._get_request_payload = _patched.__get__(llm, type(llm))

    return llm


class BaseLLM():
    """
    LLM initialization
    """
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config


class ThisIsModel(BaseLLM):
    """
    LLM/VLM initialization
    """
    def create_client(self):
        try:
            thinking = self.model_config.get('thinking', 'disabled')
            base_url = self.model_config.get('base_url', '')

            # thinking 配置处理：
            # - 'none'   : 不传 thinking 字段（适合不支持该参数的模型，如 qwen-vl-max）
            # - 'disabled': 显式关闭（基础实例默认状态）
            #               want_tools=False+enabled 时由 base_agent bind 覆盖
            # - 'enabled': 基础实例用 enabled，但 want_tools=True 时
            #              base_agent 会 bind 覆盖为 disabled（保证多轮 tool call 安全）
            # 厂商差异由 _build_thinking_extra_body 处理，无需手动区分
            vendor = _detect_vendor(base_url)
            if thinking == 'none':
                extra_body = None
            else:
                extra_body = _build_thinking_extra_body(thinking, vendor)

            client = _create_chat_openai(
                model=self.model_config['model'],
                api_key=self.model_config['api_key'],
                base_url=base_url,
                temperature=self.model_config['temperature'],
                max_tokens=self.model_config['max_tokens'],
                streaming=self.model_config.get('stream', False),
                extra_body=extra_body,
            )
            return client
        except KeyError as e:
            error_message = f"Configuration key missing: {str(e)}"
            logging.error(f"LLM Client creation failed: {error_message}")
            raise ValueError(f"LLM Client creation failed: {error_message}") from e
        except Exception as e:
            logging.error(f"LLM Client creation failed: {str(e)}")
            raise RuntimeError(f"LLM Client creation failed: {str(e)}") from e
