import time
from typing import Any
from typing import List

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel, Field, field_validator

from core.router import Router
from core.tracing import Tracer
from core.types import ChatResult, Message, ModelMetrics, RouteDecision
from tools.local_kb_tool import search_local_kb
from tools.wikipedia_tool import wikipedia_search


class WikipediaArgs(BaseModel):
    query: str = Field(..., description="Termo de busca")

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, v: Any) -> str:
        if isinstance(v, dict):
            return v.get("query") or v.get("description") or str(v)
        return str(v).strip()


class LocalKBArgs(BaseModel):
    query: str = Field(..., description="Termo de busca")

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, v: Any) -> str:
        if isinstance(v, dict):
            return v.get("query") or v.get("description") or str(v)
        return str(v).strip()


def wikipedia_search_pt(query) -> str:
    # blindagem contra o modelo mandar dict em vez de string
    if isinstance(query, dict):
        query = query.get("query") or query.get("description") or str(query)

    query = str(query).strip()
    return wikipedia_search(query=query, lang="pt", limit=3)


def local_kb_search_simple(query) -> str:
    # blindagem contra o modelo mandar dict em vez de string
    if isinstance(query, dict):
        query = query.get("query") or query.get("description") or str(query)

    query = str(query).strip()
    return search_local_kb(query=query, max_results=3)


class OrchestratorLangChain:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        model_provider: str = "ollama",
        region_name: str = "us-east-1",
        temperature: float = 0.0,
        max_tokens: int = 800,
    ):
        self.router = Router()

        # LLM via LangChain wrapper
        if model_provider == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
            )
        elif model_provider == "bedrock":
            self.llm = ChatBedrockConverse(
                model=model_name,
                region_name=region_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(
                f"Provedor de modelo não suportado: {model_provider}"
            )


        # Tool definition
        self.tools = [
            StructuredTool.from_function(
                name="wikipedia_search",
                description=(
                    "Use SOMENTE para fatos públicos e gerais do mundo, como pessoas, países, datas, história e conceitos públicos."
                ),
                func=wikipedia_search_pt,
                args_schema=WikipediaArgs,
            ),
            StructuredTool.from_function(
                name="search_local_kb",
                description=(
                    "Use para buscar informações nos arquivos locais do projeto e da base interna, "
                    "como SRE, agentes, balanço, open finance e outros conteúdos locais."
                ),
                func=local_kb_search_simple,
                args_schema=LocalKBArgs,
            ),
        ]

        self.tool_map = {
            "wikipedia_search": wikipedia_search_pt,
            "search_local_kb": local_kb_search_simple,
        }

        system_prompt = """
        Você é um assistente útil e direto.

        Regras:
        - Use wikipedia_search apenas para conhecimento público e geral.
        - Use search_local_kb para perguntas sobre arquivos locais, documentos internos, SRE, agentes, balanço e conteúdos do projeto.
        - Não use ferramentas para cumprimentos ou small talk.
        - Se puder responder diretamente com segurança, responda sem ferramenta.
        - Se não souber, diga "não sei".
        - Ao chamar ferramentas, passe apenas o valor do argumento query como texto simples.
        - Nunca passe objetos, descrições de schema ou metadados como argumento.
        - Quando a pergunta mencionar "neste projeto", "nesse projeto", "meu projeto", "meu sistema" ou equivalente, priorize search_local_kb como fonte principal antes de responder.
        - Em perguntas comparativas sobre a arquitetura, implementação ou uso das ferramentas no projeto, use search_local_kb para buscar contexto do projeto antes de concluir.
        - Não responda de forma genérica se a pergunta estiver claramente ancorada no projeto local.
        - Para perguntas sobre como o sistema funciona (arquitetura, router, tools, decisão entre LLM e tools), formule consultas mais técnicas ao usar search_local_kb.
        - Use termos como: router, tool calling, agent, llm, kb, arquitetura, decisão.
        - Evite consultas genéricas como "como funciona o projeto".
        """

        # Prompt base do agente
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Criação do agente
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Imprime no console
            return_intermediate_steps=True,  # importante para trace
            max_iterations=4,
        )

    def run(self, messages: List[Message], user_text: str) -> ChatResult:
        start = time.perf_counter()

        tracer = Tracer()
        tracer.add("user_input", text=user_text)

        # Converte histórico para formato LangChain
        chat_history = []
        for m in messages:
            if m.role == "system":
                continue
            chat_history.append({"role": m.role, "content": m.content})

        decision = self.router.route(user_text)

        tracer.add(
            "router_decision",
            route=decision.route,
            reason=decision.reason,
            tool_name=decision.tool_name,
            tool_input=decision.tool_input,
            metadata=decision.metadata,
        )

        if decision.route == "direct_response":
            latency_ms = (time.perf_counter() - start) * 1000
            tracer.add("final_answer", text=decision.direct_response)
            return ChatResult(
                answer=decision.direct_response,
                metrics=ModelMetrics(latency_ms=latency_ms),
                trace=tracer.events,
            )

        if decision.route == "tool":
            answer = self._run_forced_tool(decision)
            tracer.add("forced_tool_call",
                       tool=decision.tool_name,
                       tool_input=decision.tool_input,
                       tool_output_preview=str(answer)[:300],
                       )
            tracer.add("final_answer", text=answer)

            # Metricas

            latency_ms = (time.perf_counter() - start) * 1000
            metrics = ModelMetrics(latency_ms=latency_ms)


            return ChatResult(
                answer=answer,
                metrics=metrics,
                trace=tracer.events,
            )

        latency_ms = 0.0

        # Execução do agente
        try:
            result = self.agent_executor.invoke(
                {
                    "input": user_text,
                    "chat_history": chat_history,
                }
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            tracer.add("agent_error", error=str(e))
            return ChatResult(
                answer="Não consegui processar sua pergunta agora.",
                metrics=ModelMetrics(latency_ms=latency_ms),
                trace=tracer.events,
            )

        answer = result["output"]
        tracer.add("final_answer", text=answer)

        # Captura steps intermediários (tool calls)
        intermediate = result.get("intermediate_steps", [])
        for step in intermediate:
            action, tool_output = step
            tracer.add(
                event="tool_call",
                tool=action.tool,
                tool_input=action.tool_input,
                tool_output_preview=str(tool_output)[:300]
            )
        latency_ms = (time.perf_counter() - start) * 1000

        # Metricas
        metrics = ModelMetrics(latency_ms=latency_ms)

        return ChatResult(
            answer=answer,
            metrics=metrics,
            trace=tracer.events,
        )

    def _run_forced_tool(self, decision: RouteDecision) -> str:
        if not decision.tool_name:
            return "Não sei."

        tool_fn = self.tool_map.get(decision.tool_name)
        if not tool_fn:
            return "Não sei."

        query = ""
        if decision.tool_input:
            query = decision.tool_input.get("query", "")
            query = self._rewrite_query(query)

        try:
            return tool_fn(query)
        except Exception as e:
            return f"Erro ao executar a ferramenta {decision.tool_name}: {str(e)}"

    def _rewrite_query(self, user_input: str) -> str:
        keywords = [
            "router", "tool", "llm", "agent",
            "kb", "arquitetura", "tool calling"
        ]
        return f"{user_input} {' '.join(keywords)}"
