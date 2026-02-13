import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from core.tyoes import Message
from core.orchestrator_langchain import OrchestratorLangChain
from core.orchestrator import detect_tool, run

# ------------------ Streamlit ------------------

orch = OrchestratorLangChain()

st.set_page_config(page_title="Chatbot + Tools + Metrics", layout="wide")
st.title("Chatbot (Ollama) com Tools, Métricas e Trace")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Você é um assistente útil. Se não souber, diga 'não sei'. Não invente fatos."}
    ]

if "trace" not in st.session_state:
    st.session_state.trace = []

col_chat, col_debug = st.columns([2, 1])

with col_chat:
    st.subheader("Chat")
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        st.markdown(f"**{m['role'].capitalize()}:** {m['content']}")

    user = st.text_input("Digite sua mensagem (use /wiki <termo> para buscar na Wikipedia):", "")
    if st.button("Enviar") and user.strip():

        with st.spinner("Pensando..."):
            try:
                # 1) tool router
                tool_name, tool_arg = detect_tool(user)
                st.session_state.trace.append({"event": "user_input", "text": user})

                # result = run(tool_name, tool_arg, user_text=user, session_messages=st.session_state.messages,
                #              session_traces=st.session_state.trace)

                result = orch.run(messages=st.session_state.messages, user_text=user.strip())


            except Exception as e:
                st.error(f"Erro: {e}")
                st.stop()

        st.session_state.messages.append({"role":"user", "content":user.strip()})
        st.session_state.messages.append({"role": "assistant", "content": result.answer})

        st.session_state.trace = result.trace
        st.session_state.metrics = result.metrics.__dict__

        st.rerun()

with col_debug:
    st.subheader("Métricas & Trace")
    st.caption("Mostra latência, tokens (aprox do Ollama) e eventos (tool calls etc).")
    st.json(st.session_state.trace[-8:])
