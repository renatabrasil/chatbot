import json
import time
from typing import List, Tuple, Dict, Any

import requests
from langchain_classic.chains.question_answering.map_reduce_prompt import messages

from core.tracing import Tracer
from core.types import Message, ModelMetrics, ChatResult

# ---- Ollama API ----
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def _ns_to_ms(ns):
    if ns is None:
        return None
    return round(ns/1_000_000, 1)

def ollama_chat(messages: List[Message]) -> Tuple[str, ModelMetrics, Dict[str, Any]]:
    print(f"Chamanndo ollamachat: {messages}\n")
    payload = {
        "model": MODEL,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
    }
    start = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    latency_ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    data = r.json()

    metrics = ModelMetrics(
        latency_ms=round(latency_ms, 1),
        input_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"),
    )

    # extra raw timings (opcional)
    raw = {
        "total_duration_ms": _ns_to_ms(data.get("total_duration")),
        "prompt_eval_duration_ms": _ns_to_ms(data.get("prompt_eval_duration")),
        "eval_duration_ms": _ns_to_ms(data.get("eval_duration")),
    }

    return data["messages"]["content"], metrics, raw

# ---- Tools (manual allowlist) ----
from tools.wikipedia_tool import wikipedia_search

TOOLS = {
    "wikipedia_search": wikipedia_search,
}

# ---- JSON Protocol ----
# Model must output ONLY one JSON object:
# {"action":"tool","tool_name":"wikipedia_search","args":{"query":"Brasil","lang":"pt","limit":3}}
# or {"action":"final","final":"..."}
SYSTEM_JSON_POLICY = """Você é um orquestrador de ferramentas.

Você DEVE responder SOMENTE com um JSON válido (um único objeto), sem texto fora do JSON.

Escolha UMA ação:

1) Para usar ferramenta:
{"action":"tool","tool_name":"<nome>","args":{...}}

2) Para responder ao usuário:
{"action":"final","final":"<resposta em português>"}

Regras IMPORTANTES:
- NÃO use ferramentas para cumprimentos ("oi", "olá"), small talk, agradecimentos.
- Use ferramentas SOMENTE quando precisar de fatos externos verificáveis.
- Ferramentas permitidas: ["wikipedia_search"].
- Se não tiver certeza, responda com {"action":"final","final":"não sei"}.
"""

def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Strict JSON parse. If model returns extra text, try to salvage by extracting the first {...}.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # salvage: find first JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
        raise

def toolcall_message(tool_name: str, tool_output: str) -> Message:
    # Represent tool output back to the model in a consistent way
    return Message(
        role="assistant",
        content=f"[tool:{tool_name} output]\n{tool_output}",
    )


class OrchestratorManualToolCalling:
    def __init__(self, max_steps: int = 3):
        self.max_steps = max_steps

    def run(self, messages: List[Message], user_text: str) -> ChatResult:
        print(f"Chamanndo run: {messages}\n")
        tracer = Tracer()
        tracer.add("user_input", text=user_text)

        # Build working message list: keep original messages + policy system message
        working: List[Message] = []

        # Keep original system messages (if any) but put JSON policy on top
        working.append(Message(role="system", content=SYSTEM_JSON_POLICY))

        # Add previous conversation (excluding old system if you prefer)
        for m in messages:
            if m['role'] == "system":
                continue
            working.append(m)

        # Add current user input
        working.append(Message(role="user", content=user_text))

        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0

        for step in range(1, self.max_steps + 1):
            tracer.add("step_start", step=step)

            # 1) call model
            model_text, mm, raw = ollama_chat(working)
            total_latency_ms += mm.latency_ms
            if mm.input_tokens:
                total_input_tokens += mm.input_tokens
            if mm.output_tokens:
                total_output_tokens += mm.output_tokens

            tracer.add(
                "llm_response_raw",
                step=step,
                preview=model_text[:300],
                latency_ms=mm.latency_ms,
                input_tokens=mm.input_tokens,
                output_tokens=mm.output_tokens,
                **raw,
            )

            # 2) parse JSON
            try:
                obj = parse_json_strict(model_text)
            except Exception as e:
                tracer.add("json_parse_error", step=step, error=str(e))
                # fallback: ask model to repair once, otherwise finalize
                repair_prompt = (
                    "Você respondeu fora do formato. Responda SOMENTE com JSON válido "
                    "seguindo exatamente o protocolo. Sem texto extra."
                )
                working.append(Message(role="assistant", content=repair_prompt))
                continue

            action = obj.get("action")
            if action == "final":
                final_text = obj.get("final", "não sei")
                tracer.add("final", step=step, final_preview=final_text[:300])
                metrics = ModelMetrics(
                    latency_ms=round(total_latency_ms, 1),
                    input_tokens=total_input_tokens or None,
                    output_tokens=total_output_tokens or None,
                )
                return ChatResult(answer=final_text, metrics=metrics, trace=tracer.events)

            if action == "tool":
                tool_name = obj.get("tool_name")
                args = obj.get("args", {}) or {}
                if tool_name not in TOOLS:
                    tracer.add("tool_blocked", step=step, tool=tool_name)
                    # force final
                    working.append(
                        Message(
                            role="assistant",
                            content='{"action":"final","final":"não posso usar essa ferramenta."}',
                        )
                    )
                    continue

            tracer.add("tool_call", step=step, tool=tool_name, args=args)

            # 3) run tool
            t0 = time.perf_counter()
            try:
                out = TOOLS[tool_name](**args) if isinstance(args, dict) else TOOLS[tool_name](str(args))
                ok = True
            except Exception as e:
                out = f"ERROR: {e}"
                ok = False
            tool_ms = (time.perf_counter() - t0) * 1000
            tracer.add("tool_result", step=step, tool=tool_name, ok=ok, tool_latency_ms=round(tool_ms, 1),
                       preview=str(out)[:300])

            # 4) feed tool output back to model and continue
            working.append(toolcall_message(tool_name, str(out)))
            # Ask model to produce final OR another tool call
            working.append(
                Message(
                    role="assistant",
                    content="Agora, com base no output da ferramenta, responda seguindo o protocolo JSON.",
                )
            )
            continue

            tracer.add("unknown_action", step=step, obj=obj)

        # If max steps reached
        tracer.add("max_steps_reached", max_steps=self.max_steps)
        metrics = ModelMetrics(
            latency_ms=round(total_latency_ms, 1),
            input_tokens=total_input_tokens or None,
            output_tokens=total_output_tokens or None,
        )
        return ChatResult(answer="Não consegui concluir com segurança.", metrics=metrics, trace=tracer.events)
