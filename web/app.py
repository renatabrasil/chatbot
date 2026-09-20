import sys
from pathlib import Path

import streamlit as st



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.bedrock_provider import BedrockProvider
from core.orchestrator_langchain import OrchestratorLangChain
from core.orchestrator import Orchestrator
from core.types import Message
from providers.ollama_provider import OllamaProvider

st.set_page_config(page_title="GenAI Web (Híbrido)", layout="wide")
st.title("Chatbot Web (Streamlit) — camadas separadas")





USE_LANGCHAIN = True

if USE_LANGCHAIN:
    orch = OrchestratorLangChain(model_name="amazon.nova-lite-v1:0", model_provider="bedrock")
else:
    # provider = OllamaProvider(model="llama3.2:3b")
    provider = BedrockProvider(
        # nova lite: amazon.nova-lite-v1:0
        model_id="amazon.nova-lite-v1:0",
        # model_id="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="us-east-1",
    )
    orch = Orchestrator(provider)

if "messages" not in st.session_state:
    st.session_state.messages = [
        Message(role="system", content="Você é um assistente útil e direto. Se não souber, diga 'não sei'. Não invente fatos.")
    ]

if "trace" not in st.session_state:
    st.session_state.trace = []

if "metrics" not in st.session_state:
    st.session_state.metrics = None

col_chat, col_debug = st.columns([2,1])


with col_chat:
    st.subheader("Chat")
    for m in st.session_state.messages:
        if m.role == "system":
            continue
        st.markdown(f"**{m.role.capitalize()}:** {m.content}")

    user_text = st.text_input(label="Mensagem:", value="")
    if st.button("Enviar") and user_text.strip():
        result = orch.run(messages=st.session_state.messages, user_text=user_text.strip())
        # atualiza conversa
        st.session_state.messages.append(Message(role="user", content=user_text.strip()))
        st.session_state.messages.append(Message(role="assistant", content=result.answer))
        # guarda trace (último request)
        st.session_state.trace = result.trace

        st.session_state.metrics = result.metrics

        st.rerun()


with col_debug:
    st.subheader("Métricas")
    metrics = st.session_state.metrics
    st.json({
        "latency_ms": metrics.latency_ms if metrics else None,
        "prompt_tokens": metrics.prompt_tokens if metrics else None,
        "completion_tokens": metrics.completion_tokens if metrics else None,
        "total_duration_ms": metrics.total_duration_ms if metrics else None,
    })

    st.subheader("Trace (execução)")
    if st.session_state.trace:
        st.json([{"event": e.event, **e.data} for e in st.session_state.trace][-12:])
    else:
        st.caption("Nenhum trace ainda")