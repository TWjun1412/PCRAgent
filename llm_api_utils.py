"""
LLM API call error formatting and printing (Connection error, etc.)
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

# Default read timeout (seconds) for long text denoising/quality check; original Arbiter had only 30s易 ReadTimeout
DEFAULT_API_TIMEOUT = 180.0
DEFAULT_API_CONNECT_TIMEOUT = 30.0
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 16384
# Chinese: 1 character ≈ 1~1.5 tokens; for long text rewriting, output tokens are scaled by character count
DEFAULT_OUTPUT_TOKEN_MULTIPLIER = 2.5


def estimate_max_tokens(
    *text_parts: str,
    config: Optional[Dict[str, Any]] = None,
    floor: int = 256,
    multiplier: Optional[float] = None,
) -> int:
    """
    Estimate max_tokens for completion based on input text length to prevent long text output truncation.
    """
    cfg = config or {}
    ceiling = int(cfg.get("llm_max_output_tokens", DEFAULT_LLM_MAX_OUTPUT_TOKENS))
    mult = multiplier if multiplier is not None else float(
        cfg.get("llm_output_token_multiplier", DEFAULT_OUTPUT_TOKEN_MULTIPLIER)
    )
    max_chars = max((len(t) for t in text_parts if t), default=0)
    total_chars = sum(len(t) for t in text_parts if t)
    # Rewrite tasks: output length should cover the longest input
    basis = max(max_chars, total_chars // 2)
    estimated = int(basis * mult) + 512
    return max(floor, min(ceiling, estimated))


def warn_if_output_truncated(response: Any, *, context: str = "") -> None:
    """If finish_reason is length, warn that output may be truncated."""
    try:
        reason = response.choices[0].finish_reason
        if reason == "length":
            print(
                f"【Output may be truncated】: {context or 'LLM'} | "
                f"finish_reason=length — Increase llm_max_output_tokens in config.json",
                flush=True,
            )
    except Exception:
        pass


def completion_kwargs_with_max_tokens(
    messages: list,
    *,
    max_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """If max_tokens is not specified, estimate based on message content automatically."""
    out = dict(kwargs)
    if max_tokens is None or max_tokens <= 0:
        parts = [m.get("content", "") or "" for m in messages if isinstance(m, dict)]
        out["max_tokens"] = estimate_max_tokens(*parts, config=config)
    else:
        out["max_tokens"] = max_tokens
    return out


def _exc_chain(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    while cause:
        parts.append(f"  └─ {type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return "\n".join(parts)


def _response_hint(exc: BaseException) -> str:
    """Extract HTTP status, body, etc. from OpenAI SDK exception."""
    hints = []
    status = getattr(exc, "status_code", None)
    if status is not None:
        hints.append(f"HTTP status code: {status}")
    body = getattr(exc, "body", None)
    if body is not None:
        hints.append(f"Response body: {body}")
    response = getattr(exc, "response", None)
    if response is not None:
        hints.append(f"response: {response}")
    code = getattr(exc, "code", None)
    if code is not None:
        hints.append(f"Error code: {code}")
    return "\n".join(hints) if hints else ""


def format_api_error(
    exc: BaseException,
    *,
    context: str = "",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> str:
    lines = ["=" * 56, "【LLM API call failed】"]
    if context:
        lines.append(f"Context: {context}")
    if model:
        lines.append(f"Model: {model}")
    if base_url:
        lines.append(f"API base_url: {base_url}")
    if attempt is not None and max_attempts is not None:
        lines.append(f"Retry: {attempt}/{max_attempts} times")
    lines.append("-" * 56)
    lines.append(_exc_chain(exc))
    hint = _response_hint(exc)
    if hint:
        lines.append("-" * 56)
        lines.append(hint)
    msg = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    if "timeout" in msg or "timeout" in exc_name:
        lines.append("-" * 56)
        lines.append(
            "Common reasons: text too long or model response slow, api_timeout too small, network latency high. "
            "Increase api_timeout in config.json (recommended 180–300)."
        )
    elif "connection" in msg or "connect" in exc_name:
        lines.append("-" * 56)
        lines.append(
            "Common reasons: network unreachable, proxy/VPN, base_url error, firewall blocked, "
            "API service down or local DNS resolution failed."
        )
    lines.append("=" * 56)
    return "\n".join(lines)


def resolve_api_timeout(
    timeout: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    if timeout is not None:
        return float(timeout)
    if config and config.get("api_timeout") is not None:
        return float(config["api_timeout"])
    return DEFAULT_API_TIMEOUT


def resolve_connect_timeout(config: Optional[Dict[str, Any]] = None) -> float:
    if config and config.get("api_timeout_connect") is not None:
        return float(config["api_timeout_connect"])
    return DEFAULT_API_CONNECT_TIMEOUT


def create_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    *,
    timeout: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create an OpenAI client with reasonable read timeout (long medical dialogue recommended api_timeout>=180)."""
    from openai import OpenAI

    read_timeout = resolve_api_timeout(timeout, config)
    connect_timeout = resolve_connect_timeout(config)
    base = (base_url or (config or {}).get("base_url") or "https://api.chatanywhere.tech/v1").rstrip("/")

    try:
        import httpx

        timeout_obj = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=60.0,
            pool=30.0,
        )
    except ImportError:
        timeout_obj = read_timeout

    return OpenAI(api_key=api_key, base_url=base, timeout=timeout_obj)


def print_api_error(
    exc: BaseException,
    *,
    context: str = "",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    show_traceback: bool = False,
) -> str:
    """Print detailed error information to the console and return a formatted string."""
    text = format_api_error(
        exc,
        context=context,
        model=model,
        base_url=base_url,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    print(text, flush=True)
    if show_traceback:
        print("【Stack Trace】", flush=True)
        traceback.print_exc()
    return text


def chat_completions_create(
    client: Any,
    *,
    context: str = "",
    model: str,
    messages: list,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    show_traceback_on_final: bool = True,
    max_tokens: Optional[int] = None,
    api_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """
    Call the OpenAI API to generate a chat completion.
    """
    import time

    try:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            RateLimitError,
        )
    except ImportError:
        APIConnectionError = APIError = RateLimitError = APITimeoutError = Exception  # type: ignore

    retriable = (APIConnectionError, RateLimitError, APITimeoutError)

    base_url = getattr(client, "base_url", None) or getattr(
        getattr(client, "_client", None), "base_url", None
    )
    if base_url is not None:
        base_url = str(base_url)

    create_kwargs = completion_kwargs_with_max_tokens(
        messages,
        max_tokens=max_tokens,
        config=api_config,
        **kwargs,
    )

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                **create_kwargs,
            )
            warn_if_output_truncated(resp, context=context)
            return resp
        except retriable as e:
            last_exc = e
            print_api_error(
                e,
                context=context,
                model=model,
                base_url=base_url,
                attempt=attempt,
                max_attempts=max_retries,
            )
            if attempt < max_retries:
                delay = initial_delay * (backoff_factor ** (attempt - 1))
                if isinstance(e, APITimeoutError):
                    delay = max(delay, 5.0)
                print(f"[{context}] {delay:.0f} seconds later retry…", flush=True)
                time.sleep(delay)
            else:
                if show_traceback_on_final:
                    traceback.print_exc()
        except APIError as e:
            print_api_error(
                e,
                context=context,
                model=model,
                base_url=base_url,
                show_traceback=show_traceback_on_final,
            )
            raise
        except Exception as e:
            err_text = str(e).lower()
            if "connection" in err_text or "timeout" in err_text or "timed out" in err_text:
                last_exc = e
                print_api_error(
                    e,
                    context=context,
                    model=model,
                    base_url=base_url,
                    attempt=attempt,
                    max_attempts=max_retries,
                )
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** (attempt - 1))
                    if "timeout" in err_text or "timed out" in err_text:
                        delay = max(delay, 5.0)
                    print(f"[{context}] {delay:.0f} seconds later retry…", flush=True)
                    time.sleep(delay)
                else:
                    if show_traceback_on_final:
                        traceback.print_exc()
            else:
                print_api_error(
                    e,
                    context=context,
                    model=model,
                    base_url=base_url,
                    show_traceback=show_traceback_on_final,
                )
                raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[{context}] chat.completions.create failed and no exception information")
