import time
import json
import requests
import streamlit as st
from streamlit import metric

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

# ------------------ Tool: Wikipedia (MediaWiki API) ------------------
def wikipedia_search(query: str, limit: int = 3) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = data.get("query", {}).get("search", [])
    if not results:
        return "No results."
    lines = []
    for i, it in enumerate(results, 1):
        title = it.get("title", "")
        snippet = it.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        lines.append(f"{i}. {title} - {snippet}")
    return "\n".join(lines)

# ------------------ Simple router (nao-LLM) ------------------
def detect_tool(user_text: str):
    t = user_text.strip().lower()
    if t.startswith("/wiki"):
        return "wikipedia_search", user_text[len("/wiki "):].strip()
    return None, None

# ------------------ Ollama call ------------------

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

# ------------------ Streamlit ------------------

st.set_page_config(page_title="Chatbot + Tools + Metrics", layout="wide")
st.title("Chatbot (Ollama) com Tools, Métricas e Trace")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Você é um assistente útil. Se não souber, diga 'não sei'. Não invente fatos."}
    ]

if "trace" not in st.session_state:
    st.session_state.trace = []

col_chat, col_debug = st.columns([2,1])

with col_chat:
    st.subheader("Chat")
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        st.markdown(f"**{m['role'].capitalize()}:** {m['content']}")

    user = st.text_input("Digite sua mensagem (use /wiki <termo> para buscar na Wikipedia):", "")
    if st.button("Enviar") and user.strip():
        # 1) tool router
        tool_name, tool_arg = detect_tool(user)
        st.session_state.trace.append({"event": "user_input", "text": user})

        if tool_name == "wikipedia_search":
            st.session_state.trace.append({"event": "tool_call", "tool": tool_name})
            tool_out = wikipedia_search(tool_arg)
            st.session_state.trace.append({"event": "tool_result", "tool": tool_name})

            # injeta tool output no chat como contexto
            st.session_state.messages.append({"role": "user", "content": user})
            st.session_state.messages.append({"role": "assistant", "content": f"Resultado do /wiki:\n{tool_out}\n\nO que você quer fazer com isso?"})
        else:
            st.session_state.messages.append({"role": "user", "content": user})
            answer, metrics = ollama_chat(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.trace.append({"event": "model_response", "metrics": metrics})

        st.rerun()

with col_debug:
    st.subheader("Métricas & Trace")
    st.caption("Mostra latência, tokens (aprox do Ollama) e eventos (tool calls etc).")
    st.json(st.session_state.trace[-8:])