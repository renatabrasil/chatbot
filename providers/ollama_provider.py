import time
from typing import List

import requests

from core.types import Message, ModelMetrics
from providers.base import LLMProvider

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def _ns_to_ms(ns):
    # return round(ns/1_000_000, 1) if ns else None:
    # return (round(ns/1_000_000, 1), None) [ns]
    # return {True: round(ns/1_000_000, 1), False: None}[ns]
    if ns is None:
        return None
    return round(ns / 1_000_000, 1)


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = MODEL, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/chat"

    def chat(self, messages: List[Message]) -> tuple[str, ModelMetrics]:
        """
                Returns chats content provided by LLM and associated metrics
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }

        start = time.perf_counter()
        r = requests.post(url=self.url, json=payload, timeout=120)
        latency_ms = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        data = r.json()

        metrics = ModelMetrics(
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_duration_ms=_ns_to_ms(data.get("total_duration")),
            prompt_eval_duration_ms=_ns_to_ms(data.get("prompt_eval_duration")),
            eval_duration_ms=_ns_to_ms(data.get("eval_duration")),
        )

        return data["message"]["content"], metrics
