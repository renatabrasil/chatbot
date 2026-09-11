from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None  # útil pra tools


@dataclass
class ModelMetrics:
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_duration_ms: Optional[float] = None
    prompt_eval_duration_ms: Optional[float] = None
    eval_duration_ms: Optional[float] = None


@dataclass
class TraceEvent:
    event: str
    data: Dict[str, Any]

@dataclass
class ChatResult:
    answer: str
    metrics: ModelMetrics
    trace: List[TraceEvent]


@dataclass
class RouteDecision:
    route: str  # "direct_response" | "tool" | "agent"
    reason: str
    direct_response: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)