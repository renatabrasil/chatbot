from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ModelMetrics:
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ChatResult:
    answer: str
    metrics: ModelMetrics
    trace: List[Dict[str, Any]]
