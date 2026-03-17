from __future__ import annotations

from threading import Lock

_lock = Lock()
_state: dict[str, float | int] = {
    "request_count": 0,
    "request_latency_ms_total": 0.0,
    "llm_call_count": 0,
    "llm_latency_ms_total": 0.0,
    "llm_fallback_count": 0,
    "llm_estimated_tokens_total": 0,
}


def observe_request(duration_ms: float) -> None:
    with _lock:
        _state["request_count"] += 1
        _state["request_latency_ms_total"] += duration_ms


def observe_llm_call(duration_ms: float, *, used_fallback: bool, estimated_tokens: int) -> None:
    with _lock:
        _state["llm_call_count"] += 1
        _state["llm_latency_ms_total"] += duration_ms
        _state["llm_estimated_tokens_total"] += estimated_tokens
        if used_fallback:
            _state["llm_fallback_count"] += 1


def snapshot() -> dict[str, float | int]:
    with _lock:
        request_count = int(_state["request_count"])
        llm_call_count = int(_state["llm_call_count"])
        return {
            **_state,
            "request_latency_ms_avg": round(float(_state["request_latency_ms_total"]) / max(1, request_count), 2),
            "llm_latency_ms_avg": round(float(_state["llm_latency_ms_total"]) / max(1, llm_call_count), 2),
        }
