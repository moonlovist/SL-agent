"""
continuation.py — shared truncation-detection and continuation-loop logic.

Used by single_hypothesis, hypothesis_synthesis, and AnalysisAuditor to handle
LLM output truncation (finish_reason=length).  Supports both streaming
(agent.astream) and non-streaming (agent.ainvoke / agent.invoke) paths,
with automatic retry when the continuation itself is truncated.
"""

import json
import logging
from typing import Callable, Optional

from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Shared helpers (single source of truth — previously duplicated in 3 files)
# ---------------------------------------------------------------------------

def _find_last_ai_message(messages: list):
    """Return the last AI message, skipping tool/other message types."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            return msg
    return None


def _find_last_content_ai_message(messages: list):
    """Return the last AI message that has non-empty text content.

    When the agent makes tool calls, the final AI message may be a
    tool-call-only message with empty ``content``.  This helper skips
    those and returns the last AI message that actually contains text
    (e.g., the truncated JSON output that needs continuation).
    """
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai":
            content = getattr(msg, "content", None)
            if content and isinstance(content, str) and content.strip():
                return msg
    return None


def _is_truncated(messages: list, *, since: int = 0) -> bool:
    """Check whether any AI message was truncated by max_tokens.

    Scans ALL AI messages (not just the last one) because in multi-turn
    tool-calling agents, truncation may happen in an intermediate turn
    while the final turn completes normally (e.g., a tool-only final
    message with finish_reason=stop).

    Parameters
    ----------
    since : int
        Only check messages from this index onward.  Used by continuation
        loops to avoid re-detecting truncation markers from the original
        (already-handled) run.
    """
    for msg in reversed(messages[since:]):
        if getattr(msg, "type", None) == "ai":
            if msg.response_metadata.get("finish_reason") == "length":
                return True
    return False


def _format_tool_call(tc) -> str:
    """Format a single tool call as readable markdown."""
    name = tc.get("name", "unknown")
    args = tc.get("args", {})
    args_json = json.dumps(args, indent=2, ensure_ascii=False)
    return f"**`{name}`**\n```json\n{args_json}\n```"


def _format_tool_result(msg) -> str:
    """Format a tool result message as readable markdown."""
    content = msg.content if hasattr(msg, "content") else str(msg)
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        return f"```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
    except (json.JSONDecodeError, TypeError):
        return str(content)


# ---------------------------------------------------------------------------
# Terse retry prompt (used on 2nd+ continuation attempts)
# ---------------------------------------------------------------------------

_TERSE_RETRY_PROMPT = (
    "STILL truncated. Continue DIRECTLY from the last "
    "character. Do NOT repeat or explain — output ONLY "
    "the remaining content."
)


# ---------------------------------------------------------------------------
# Streaming continuation loop
# ---------------------------------------------------------------------------

async def run_continuation_streaming(
    agent,
    accumulated_messages: list,
    *,
    config: dict,
    stream_md_path: Optional[str] = None,
    continuation_prompt: str = "",
    max_attempts: int = 3,
    log_prefix: str = "",
) -> list:
    """Run the truncation-continuation loop using ``agent.astream``.

    Appends all continuation messages (AI + tool) to *accumulated_messages*
    in-place.  When *stream_md_path* is set, writes turn-by-turn markdown
    including tool calls and results.

    Parameters
    ----------
    agent : langchain agent
    accumulated_messages : list
        Live message list from the original streaming run (mutated in-place).
    config : dict
        Agent config (recursion_limit, etc.).
    stream_md_path : str or None
        If set, append continuation output to this file.
    continuation_prompt : str
        Prompt for the first continuation attempt.  On 2nd+ attempts a
        terse retry prompt is used automatically.
    max_attempts : int
        Maximum number of continuation attempts (default 3).
    log_prefix : str
        Prefix for log messages (e.g. "FeatureAuditor").

    Returns
    -------
    list
        The mutated *accumulated_messages* (same object).
    """
    ctn_attempt = 0
    n_before = len(accumulated_messages)
    while _is_truncated(accumulated_messages, since=n_before) and ctn_attempt < max_attempts:
        ctn_attempt += 1
        logging.warning(
            f"{log_prefix} output truncated (finish_reason=length), "
            f"continuation attempt {ctn_attempt}/{max_attempts}..."
        )
        _ctn_md = None
        try:
            n_this_attempt = len(accumulated_messages)  # snapshot before streaming
            if stream_md_path:
                _ctn_md = open(stream_md_path, "a", encoding="utf-8")
                _ctn_md.write(f"\n---\n### 🔄 Continuation {ctn_attempt}\n\n")

            continuation_msgs = list(accumulated_messages)
            prompt = continuation_prompt if ctn_attempt == 1 else _TERSE_RETRY_PROMPT
            continuation_msgs.append(HumanMessage(content=prompt))

            if _ctn_md:
                _ctn_md.write(f"**Prompt**: {prompt}\n\n")
                _ctn_md.flush()

            ctn_turn = 0
            async for event in agent.astream(
                {"messages": continuation_msgs},
                config=config,
                stream_mode="updates",
            ):
                for _node_name, update in event.items():
                    for msg in update.get("messages", []):
                        msg_type = getattr(msg, "type", None)
                        if msg_type == "ai":
                            ctn_turn += 1
                            content = (
                                msg.content
                                if hasattr(msg, "content")
                                else ""
                            )
                            tool_calls = getattr(msg, "tool_calls", None)
                            lines = []
                            if content:
                                lines.append(content.strip())
                            if tool_calls:
                                for tc in tool_calls:
                                    lines.append(_format_tool_call(tc))
                            if lines and _ctn_md:
                                _ctn_md.write(
                                    f"### Assistant (ctn turn {ctn_turn})\n\n"
                                )
                                _ctn_md.write("\n\n".join(lines))
                                _ctn_md.write("\n\n")
                                _ctn_md.flush()
                        elif msg_type == "tool":
                            if _ctn_md:
                                _ctn_md.write("### Tool Result\n\n")
                                _ctn_md.write(_format_tool_result(msg))
                                _ctn_md.write("\n\n")
                                _ctn_md.flush()
                        accumulated_messages.append(msg)

            # Only check messages added by this continuation for truncation
            n_before = n_this_attempt

            if _ctn_md:
                _ctn_md.close()
                _ctn_md = None
            logging.info(
                f"{log_prefix} continuation {ctn_attempt} completed "
                f"({ctn_turn} turns)."
            )
        except Exception as exc:
            logging.warning(
                f"{log_prefix} continuation {ctn_attempt} failed: {exc}. "
                "Returning truncated result."
            )
            if _ctn_md:
                try:
                    _ctn_md.close()
                except Exception:
                    pass
            break  # don't retry on exception

    if _is_truncated(accumulated_messages, since=n_before):
        logging.error(
            f"{log_prefix} ⚠️  STILL TRUNCATED after "
            f"{ctn_attempt} continuation attempt(s) — output may be incomplete, "
            f"but pipeline will continue with partial result."
        )

    return accumulated_messages


# ---------------------------------------------------------------------------
# Non-streaming continuation loop (ainvoke)
# ---------------------------------------------------------------------------

async def run_continuation_ainvoke(
    agent,
    messages: list,
    *,
    config: dict,
    continuation_prompt: str = "",
    max_attempts: int = 3,
    log_prefix: str = "",
) -> list:
    """Run the truncation-continuation loop using ``agent.ainvoke``.

    Appends all continuation messages to *messages* in-place.

    Parameters
    ----------
    agent : langchain agent
    messages : list
        Message list from the original ``ainvoke`` run (mutated in-place).
    config : dict
        Agent config.
    continuation_prompt : str
        Prompt for the first continuation attempt.
    max_attempts : int
        Max continuation attempts (default 3).
    log_prefix : str
        Prefix for log messages.

    Returns
    -------
    list
        The mutated *messages* (same object).
    """
    ctn_attempt = 0
    n_before = len(messages)
    while _is_truncated(messages, since=n_before) and ctn_attempt < max_attempts:
        ctn_attempt += 1
        logging.warning(
            f"{log_prefix} output truncated (non-streaming), "
            f"continuation attempt {ctn_attempt}/{max_attempts}..."
        )
        try:
            continuation_msgs = list(messages)
            prompt = continuation_prompt if ctn_attempt == 1 else _TERSE_RETRY_PROMPT
            continuation_msgs.append(HumanMessage(content=prompt))
            continuation_result = await agent.ainvoke(
                {"messages": continuation_msgs}, config=config
            )
            # Only append NEW messages (not the full history) to avoid
            # duplication and spurious truncation re-detection.
            result_msgs = continuation_result.get("messages", [])
            new_msgs = result_msgs[len(continuation_msgs):]
            messages.extend(new_msgs)
            n_before = len(messages) - len(new_msgs)  # only check new msgs next round
            logging.info(
                f"{log_prefix} continuation {ctn_attempt} completed successfully."
            )
        except Exception as exc:
            logging.warning(
                f"{log_prefix} continuation {ctn_attempt} failed: {exc}. "
                "Returning truncated result."
            )
            break

    if _is_truncated(messages, since=n_before):
        logging.error(
            f"{log_prefix} ⚠️  STILL TRUNCATED after "
            f"{ctn_attempt} continuation attempt(s) — output may be incomplete, "
            f"but pipeline will continue with partial result."
        )

    return messages


# ---------------------------------------------------------------------------
# Non-streaming continuation loop (invoke — sync)
# ---------------------------------------------------------------------------

def run_continuation_invoke(
    agent,
    messages: list,
    *,
    config: dict,
    continuation_prompt: str = "",
    max_attempts: int = 3,
    log_prefix: str = "",
) -> list:
    """Run the truncation-continuation loop using ``agent.invoke`` (sync).

    Appends all continuation messages to *messages* in-place.

    Parameters
    ----------
    agent : langchain agent
    messages : list
        Message list from the original ``invoke`` run (mutated in-place).
    config : dict
        Agent config.
    continuation_prompt : str
        Prompt for the first continuation attempt.
    max_attempts : int
        Max continuation attempts (default 3).
    log_prefix : str
        Prefix for log messages.

    Returns
    -------
    list
        The mutated *messages* (same object).
    """
    ctn_attempt = 0
    n_before = len(messages)
    while _is_truncated(messages, since=n_before) and ctn_attempt < max_attempts:
        ctn_attempt += 1
        logging.warning(
            f"{log_prefix} output truncated (sync), "
            f"continuation attempt {ctn_attempt}/{max_attempts}..."
        )
        try:
            continuation_msgs = list(messages)
            prompt = continuation_prompt if ctn_attempt == 1 else _TERSE_RETRY_PROMPT
            continuation_msgs.append(HumanMessage(content=prompt))
            continuation_result = agent.invoke(
                {"messages": continuation_msgs}, config=config
            )
            # Only append NEW messages (not the full history) to avoid
            # duplication and spurious truncation re-detection.
            result_msgs = continuation_result.get("messages", [])
            new_msgs = result_msgs[len(continuation_msgs):]
            messages.extend(new_msgs)
            n_before = len(messages) - len(new_msgs)  # only check new msgs next round
            logging.info(
                f"{log_prefix} continuation {ctn_attempt} completed successfully."
            )
        except Exception as exc:
            logging.warning(
                f"{log_prefix} continuation {ctn_attempt} failed: {exc}. "
                "Returning truncated result."
            )
            break

    if _is_truncated(messages, since=n_before):
        logging.error(
            f"{log_prefix} ⚠️  STILL TRUNCATED after "
            f"{ctn_attempt} continuation attempt(s) — output may be incomplete, "
            f"but pipeline will continue with partial result."
        )

    return messages
