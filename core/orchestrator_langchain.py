from typing import List

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from core.tracing import Tracer
from core.tyoes import ChatResult, Message, ModelMetrics
from tools.wikipedia_tool import wikipedia_search

class WikipediaArgs(BaseModel):
    query: str = Field(..., description="Termo de busca na wikipedia")
    lang: str = Field("pt", description="Idioma da Wikipedia: 'pt' ou 'en'")
    limit: int = Field(3, description="Quantidade de resultados (1 a 5)")

class OrchestratorLangChain:
    def __init__(self, model_name: str = "llama3.2:3b"):
        # LLM via LangChain wrapper
        self.llm = ChatOllama(
            model = model_name,
            temperature = 0.2,
        )

        # Tool definition
        self. tools = [
            StructuredTool.from_function(
                name="wikipedia_search",
                description=(
                    "Search Wikipedia for general knowledge. "
                    "Use when the user asks about factual topics, people, places, history, etc. "
                    "You must use this tool only if user explicit requers an info."
                ),
                func=wikipedia_search,
                args_schema=WikipediaArgs,
            )
        ]

        # Prompt base do agente
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Você é um assistente útil. Use ferrametnas quando necessário. "
                 "Se não souber, diga 'não sei'. Não invente fatos. "
                 "Não chame ferramenta se não for estritamente necessário, ou seja, se vc não souber uma informação. Ex: smalltalk não precisa chamar ferramenta."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Criação do agente
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent= agent,
            tools=self.tools,
            verbose=True,   # Imprime no console
            return_intermediate_steps=True,    # importante para trace
            max_iterations=5,
        )

    def run(self, messages: List[Message], user_text: str) -> ChatResult:
        tracer = Tracer()

        tracer.add("user_input", text=user_text)

        print(f"mensagem: {messages}\n")

        # Converte histórico para formato LangChain
        chat_history = []
        for m in messages:
            if m['role'] == "system":
                continue
            chat_history.append({"role": m['role'], "content": m['content']})


        # Execução do agente
        result = self.agent_executor.invoke(
            {
                "input": user_text,
                "chat_history": chat_history,
            }
        )

        answer = result["output"]
        tracer.add("final_answer", text=answer)

        # Captura steps intermediários (tool calls)
        intermediate = result.get("intermediate_steps", [])
        for step in intermediate:
            action, tool_output = step
            tracer.add(
                "tool_call",
                tool=action.tool,
                tool_input=action.tool_input,
                tool_output_preview=str(tool_output)[:300]
            )

        # Metricas
        metrics = ModelMetrics(latency_ms=0.0)

        return ChatResult(
            answer=answer,
            metrics=metrics,
            trace=tracer.events,
        )
