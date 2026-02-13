from typing import Dict, Any

from narwhals import List

from providers.ollama_provider import ollama_chat
from tools.wikipedia_tool import wikipedia_search


def run(tool_name: str, tool_arg: str, user_text:str,  session_messages: List[Dict[str, Any]], session_traces: List[Dict[str, Any]]):
    if tool_name == "wikipedia_search":
        session_traces.append({"event": "tool_call", "tool": tool_name})
        tool_out = wikipedia_search(tool_arg)
        session_traces.append({"event": "tool_result", "tool": tool_name})

        # injeta tool output no chat como contexto
        session_messages.append({"role": "user", "content": user_text})
        session_messages.append(
            {"role": "assistant", "content": f"Resultado do /wiki:\n{tool_out}\n\nO que você quer fazer com isso?"})
    else:
        session_messages.append({"role": "user", "content": user_text})
        answer, metrics = ollama_chat(session_messages)
        session_messages.append({"role": "assistant", "content": answer})
        session_traces.append({"event": "model_response", "metrics": metrics})


def detect_tool(user_text: str):
    t = user_text.strip().lower()
    if t.startswith("/wiki"):
        return "wikipedia_search", user_text[len("/wiki "):].strip()
    return None, None