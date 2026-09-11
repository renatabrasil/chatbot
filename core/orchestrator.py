from typing import Optional, Tuple, List
from uuid import uuid4

from core.router import Router
from core.tracing import Tracer
from core.types import Message, ChatResult, ModelMetrics
from observability.logger import JsonlObservabilityLogger
from providers.base import LLMProvider
from tools.local_kb_tool import search_local_kb
from tools.wikipedia_tool import wikipedia_search


def _detect_tool(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    t = user_text.strip().lower()
    if t.startswith("/wiki"):
        return "wikipedia_search", user_text[len("/wiki "):].strip()
    return None, None


def estimate_cost_usd(model_id: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    # Placeholder.
    # Depois você troca pelos preços reais do modelo.
    if not model_id:
        return 0.0

    pricing = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input_per_1k": 0.00025,
            "output_per_1k": 0.00125,
        },
        "amazon.nova-lite-v1:0": {
            "input_per_1k": 0.00006,
            "output_per_1k": 0.00024,
        },
    }

    model_price = pricing.get(model_id)
    if not model_price:
        return 0.0

    cost = (
            (prompt_tokens / 1000) * model_price["input_per_1k"] +
            (completion_tokens / 1000) * model_price["output_per_1k"]
    )
    return round(cost, 8)


class Orchestrator:
    def __init__(self, provider: LLMProvider, obs_logger: JsonlObservabilityLogger | None = None) -> None:
        self.provider = provider
        self.router = Router()
        self.obs_logger = obs_logger or JsonlObservabilityLogger()

    def run(self, messages: List[Message], user_text: str) -> ChatResult:
        request_id = str(uuid4())
        tracer = Tracer()
        tracer.add("user_input", text=user_text)

        decision = self.router.route(user_text)

        tracer.add(
            "router_decision",
            route=decision.route,
            reason=decision.reason,
            tool_name=decision.tool_name,
            tool_input=decision.tool_input,
            metadata=decision.metadata,
        )

        model_id = getattr(self.provider, "model_id", None) or getattr(self.provider, "model", None)
        route = decision.route
        tool_name = decision.tool_name
        status = "success"

        try:
            if decision.route == "direct_response":
                answer = decision.direct_response or "Não sei."
                tracer.add("final_answer", text=answer)

                metrics = ModelMetrics(latency_ms=0.0)

                self._log_request_event(
                    request_id=request_id,
                    route=route,
                    tool_name=tool_name,
                    model_id=model_id,
                    metrics=metrics,
                    status=status,
                    user_text=user_text,
                    answer=answer,
                )

                return ChatResult(
                    answer=answer,
                    metrics=ModelMetrics(latency_ms=0.0),
                    trace=tracer.events,
                )

            if decision.route == "tool":
                answer = self._run_forced_tool(decision)
                tracer.add(
                    "forced_tool_call",
                    tool=decision.tool_name,
                    tool_input=decision.tool_input,
                    tool_output_preview=str(answer)[:300],
                )
                tracer.add("final_answer", text=answer)

                metrics = ModelMetrics(latency_ms=0.0)

                self._log_request_event(
                    request_id=request_id,
                    route=route,
                    tool_name=tool_name,
                    model_id=model_id,
                    metrics=metrics,
                    status=status,
                    user_text=user_text,
                    answer=answer,
                )

                return ChatResult(
                    answer=answer,
                    metrics=ModelMetrics(latency_ms=0.0),
                    trace=tracer.events,
                )

            # Chama o modelo (se não for para usar a ferramenta)
            messages2 = messages + [Message(role="user", content=user_text)]
            tracer.add("model_call", model=getattr(self.provider, "model", "unknown"))
            answer, metrics = self.provider.chat(messages2)
            tracer.add("model_response",
                       latency_ms=metrics.latency_ms,
                       prompt_tokens=metrics.prompt_tokens,
                       completion_tokens=metrics.completion_tokens)

            self._log_request_event(
                request_id=request_id,
                route=route,
                tool_name=tool_name,
                model_id=model_id,
                metrics=metrics,
                status=status,
                user_text=user_text,
                answer=answer,
            )

            return ChatResult(answer=answer, metrics=metrics, trace=tracer.events)
        except Exception as e:
            status = "error"

            self._log_request_event(
                request_id=request_id,
                route=route,
                tool_name=tool_name,
                model_id=model_id,
                metrics=ModelMetrics(latency_ms=0.0),
                status=status,
                user_text=user_text,
                answer="",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def _log_request_event(
            self,
            request_id: str,
            route: str,
            tool_name: str | None,
            model_id: str | None,
            metrics: ModelMetrics,
            status: str,
            user_text: str,
            answer: str,
            error_type: str | None = None,
            error_message: str | None = None,
    ) -> None:
        prompt_tokens = metrics.prompt_tokens or 0
        completion_tokens = metrics.completion_tokens or 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd = estimate_cost_usd(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        self.obs_logger.log_event({
            "environment": "dev",
            "application": "genai-web-hibrido",
            "agent_name": "chatbot_web",
            "agent_version": "v1",
            "request_id": request_id,
            "route": route,
            "tool_name": tool_name,
            "model_id": model_id,
            "latency_ms": metrics.latency_ms or 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "user_text": user_text,
            "answer_preview": answer[:300],
        })

    def _run_forced_tool(self, decision) -> str:
        if decision.tool_name == "wikipedia_search":
            query = decision.tool_input.get("query", "")
            return wikipedia_search(query=query, lang="pt")

        if decision.tool_name == "search_local_kb":
            query = decision.tool_input.get("query", "")
            return search_local_kb(query=query, max_results=3)

        return "Não sei."
