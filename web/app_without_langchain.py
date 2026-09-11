import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.orchestrator import _detect_tool
from providers.ollama_provider import ollama_chat
from tools.wikipedia_tool import wikipedia_search



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
        tool_name, tool_arg = _detect_tool(user)
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