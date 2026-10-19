import time
from typing import Optional, Tuple, List
from uuid import uuid4

import json
from pydantic import ValidationError

from core.router import Router
from core.tracing import Tracer
from core.types import Message, ChatResult, ModelMetrics, RouteDecision
from observability.logger import JsonlObservabilityLogger
from providers.base import LLMProvider
from tools.flights_tool import BUSCAR_VOOS_TOOL_SPEC, BuscarVoosArgs, InterpretacaoBuscarVoos, buscar_voos
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

    def run(self, messages: List[Message], user_text: str, modo_voos: bool = False,) -> ChatResult:
        request_id = str(uuid4())
        tracer = Tracer()
        tracer.add("user_input", text=user_text)

        if modo_voos:
            decision = RouteDecision(
                route="agent",
                reason="modo_busca_voos_controlada",
            )
        else:
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

            if modo_voos:
                answer, metrics = self._run_busca_voos_controlada(
                    messages=messages,
                    user_text=user_text,
                    tracer=tracer,
                )
            else:
                answer, metrics = self._run_agent_with_tools(
                    messages=messages,
                    user_text=user_text,
                    tracer=tracer,
                )

            tracer.add(
                "final_answer",
                text=answer,
            )

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

    def _run_agent_with_tools(
        self,
        messages: List[Message],
        user_text: str,
        tracer: Tracer,
    ) -> tuple[str, ModelMetrics]:
        start = time.perf_counter()

        system_parts = [
            m.content for m in messages if m.role == "system"
        ]
        system_parts.append(
            "Você pode consultar ofertas de voos pela ferramenta buscar_voos. "
            "Peça os campos obrigatórios que estiverem faltando. "
            "Não invente datas nem resultados de ferramentas. "
            "Informe claramente quando as ofertas forem simuladas."
            "Para cada novo pedido de busca ou atualização de voos, "
            "execute buscar_voos quando os dados necessários estiverem completos. "
            "Ofertas anteriores no histórico não substituem uma nova consulta. "
            "Se faltarem dados, pergunte antes de executar. "
            "Você pode explicar ou comparar ofertas anteriores sem consultar novamente, "
            "desde que o usuário não esteja pedindo uma nova busca. "
            "A data de volta não pode ser anterior à data de ida. "
            "Se isso acontecer, explique o problema e peça datas corrigidas. "
            "Nunca corrija ou inverta as datas por conta própria. "
            "Não apresente ofertas para um novo pedido sem receber "
            "um resultado bem-sucedido de buscar_voos nesta interação. "
            "Isso também vale para ofertas simuladas. "
        )

        # Histórico no formato nativo do Bedrock.
        working = [
            {
                "role": m.role,
                "content": [{"text": m.content}],
            }
            for m in messages
            if m.role in {"user", "assistant"}
        ]
        working.append({
            "role": "user",
            "content": [{"text": user_text}],
        })

        input_tokens = 0
        output_tokens = 0
        answer = "Não consegui concluir dentro do limite de chamadas."

        max_calls = 4

        for step in range(1, max_calls + 1):
            response = self.provider.converse(
                messages=working,
                tools=[BUSCAR_VOOS_TOOL_SPEC],
                system="\n\n".join(system_parts),
            )

            usage = response.get("usage", {})
            input_tokens += usage.get("inputTokens", 0)
            output_tokens += usage.get("outputTokens", 0)

            stop_reason = response["stopReason"]
            model_message = response["output"]["message"]

            tracer.add(
                "agent_model_response",
                step=step,
                stop_reason=stop_reason,
                prompt_tokens=usage.get("inputTokens", 0),
                completion_tokens=usage.get("outputTokens", 0),
            )

            working.append(model_message)

            if stop_reason == "end_turn":
                answer = "\n".join(
                    block["text"]
                    for block in model_message["content"]
                    if "text" in block
                ).strip()

                if not answer:
                    answer = "O modelo encerrou sem uma resposta em texto."

                break

            if stop_reason != "tool_use":
                answer = (
                    "A resposta foi interrompida antes de concluir. "
                    f"Motivo: {stop_reason}."
                )
                break

            # Não executa uma ferramenta se não houver
            # outra chamada disponível para interpretar seu resultado.
            if step == max_calls:
                tracer.add("agent_max_calls", max_calls=max_calls)
                break

            tool_results = []

            for block in model_message["content"]:
                if "toolUse" not in block:
                    continue

                pedido = block["toolUse"]
                tool_name = pedido["name"]
                tool_start = time.perf_counter()

                tracer.add(
                    "tool_call",
                    step=step,
                    tool=tool_name,
                    tool_input=pedido["input"],
                )

                status = "success"

                if tool_name != "buscar_voos":
                    status = "error"
                    result = {
                        "erro": "Ferramenta não permitida."
                    }
                else:
                    try:
                        args = BuscarVoosArgs.model_validate(
                            pedido["input"]
                        )
                        result = buscar_voos(args)

                    except ValidationError as error:
                        status = "error"
                        result = {
                            "erro": "Argumentos inválidos.",
                            "detalhes": str(error),
                        }

                    except Exception:
                        status = "error"
                        result = {
                            "erro": "Falha ao consultar voos."
                        }

                tracer.add(
                    "tool_result",
                    step=step,
                    tool=tool_name,
                    status=status,
                    tool_latency_ms=round(
                        (time.perf_counter() - tool_start) * 1000,
                        1,
                    ),
                )

                tool_results.append({
                    "toolResult": {
                        "toolUseId": pedido["toolUseId"],
                        "status": status,
                        "content": [{"json": result}],
                    }
                })

            if not tool_results:
                answer = "O modelo pediu uma ferramenta sem informar a chamada."
                break

            working.append({
                "role": "user",
                "content": tool_results,
            })

        metrics = ModelMetrics(
            latency_ms=round(
                (time.perf_counter() - start) * 1000,
                1,
            ),
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )

        return answer, metrics

    def _interpretar_busca_voos(
        self,
        messages: List[Message],
        user_text: str,
    ) -> tuple[InterpretacaoBuscarVoos, ModelMetrics]:
        schema = InterpretacaoBuscarVoos.model_json_schema()

        # Para extrair critérios, usamos apenas os pedidos do usuário.
        pedidos_anteriores = [
            m.content
            for m in messages
            if m.role == "user"
        ]

        contexto = {
            "pedidos_anteriores": pedidos_anteriores,
            "pedido_atual": user_text,
        }

        working = [{
            "role": "user",
            "content": [{
                "text": (
                    "Extraia os critérios dos dados abaixo. "
                    "O pedido atual prevalece sobre os anteriores. "
                    "Devolva apenas o objeto JSON com os campos do schema. "
                    "Não responda ao pedido de viagem nem apresente ofertas.\n\n"
                    + json.dumps(contexto, ensure_ascii=False)
                )
            }],
        }]

        system = (
            "Você extrai critérios de uma busca de voos. "
            "Responda apenas com um objeto JSON válido, "
            "sem Markdown, explicações ou tags. "
            "Não apresente ofertas nem preços. "
            "Use os critérios informados pelo usuário no histórico. "
            "A mensagem mais recente prevalece quando alterar um critério. "
            "Não use ofertas anteriores como fonte dos critérios. "
            "Para campos não informados, use null. "
            "Não invente datas, aeroportos ou quantidade de adultos. "
            "Preserve as datas informadas mesmo que estejam invertidas; "
            "a validação será feita pelo código. "
            "Siga este JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

        start = time.perf_counter()

        response = self.provider.converse(
            messages=working,
            tools=[],
            system=system,
        )

        if response["stopReason"] != "end_turn":
            raise ValueError(
                "A interpretação da busca não foi concluída."
            )

        text = "\n".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        ).strip()

        text = text.strip()

        linhas = text.splitlines()

        if (
            len(linhas) >= 3
            and linhas[0].strip().lower() in {"```json", "```"}
            and linhas[-1].strip() == "```"
        ):
            text = "\n".join(linhas[1:-1]).strip()

        # print("RESPOSTA BRUTA DA INTERPRETAÇÃO:", repr(text))
        interpretacao = InterpretacaoBuscarVoos.model_validate_json(text)
        # print("RESPOSTA BRUTA DA INTERPRETAÇÃO:", repr(text))

        usage = response.get("usage", {})

        metrics = ModelMetrics(
            latency_ms=round(
                (time.perf_counter() - start) * 1000,
                1,
            ),
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
        )

        return interpretacao, metrics

    def _run_busca_voos_controlada(
        self,
        messages: List[Message],
        user_text: str,
        tracer: Tracer,
    ) -> tuple[str, ModelMetrics]:
        start = time.perf_counter()

        interpretacao, metrics = self._interpretar_busca_voos(
            messages=messages,
            user_text=user_text,
        )

        tracer.add(
            "flight_search_interpreted",
            criterios=interpretacao.model_dump(mode="json"),
        )

        faltantes = interpretacao.campos_faltantes()

        if faltantes:
            answer = (
                "Para buscar os voos, preciso destas informações: "
                + ", ".join(faltantes)
                + "."
            )

        else:
            try:
                args = BuscarVoosArgs.model_validate(
                    interpretacao.model_dump()
                )

            except ValidationError as error:
                problemas = [
                    detalhe["msg"]
                    for detalhe in error.errors()
                ]

                tracer.add(
                    "flight_search_validation_error",
                    problemas=problemas,
                )

                answer = (
                    "Não executei a busca. "
                    + " ".join(problemas)
                    + " Pode corrigir os dados?"
                )

            else:
                tracer.add(
                    "tool_call",
                    tool="buscar_voos",
                    tool_input=args.model_dump(mode="json"),
                )

                resultado = buscar_voos(args)

                tracer.add(
                    "tool_result",
                    tool="buscar_voos",
                    status="success",
                )

                linhas = [
                    "Ofertas SIMULADAS de ida e volta "
                    f"para {args.adultos} adulto(s):"
                ]

                for oferta in resultado["ofertas"]:
                    linhas.append(
                        f"- {oferta['companhia']}: "
                        f"{resultado['moeda']} "
                        f"{oferta['preco_total']:.2f}; "
                        f"escalas na ida: {oferta['escalas_ida']}; "
                        f"escalas na volta: {oferta['escalas_volta']}; "
                        "bagagem despachada incluída: "
                        + (
                            "sim."
                            if oferta["bagagem_despachada_incluida"]
                            else "não."
                        )
                    )

                answer = "\n".join(linhas)

        metrics.latency_ms = round(
            (time.perf_counter() - start) * 1000,
            1,
        )

        return answer, metrics