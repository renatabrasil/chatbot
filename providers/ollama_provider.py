import requests
import time

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def ollama_chat(messages):
    payload = {"model": MODEL, "messages": messages, "stream": False}
    start = time.perf_counter()
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    latency_ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    data = r.json()
    # Metricas que o ollama costuma devolver
    metrics = {
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "total_duration_ms": _ns_to_ms(data.get("total_duration")),
        "eval_duration_ms": _ns_to_ms(data.get("eval_duration")),
        "prompt_eval_duration_ms": _ns_to_ms(data.get("prompt_eval_duration"))
    }
    return data["message"]["content"], metrics


def _ns_to_ms(ns):
    if ns is None:
        return None
    return round(ns/1_000_000, 1)