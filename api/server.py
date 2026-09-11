from fastapi import FastAPI
from pydantic.v1 import BaseModel

from core.orchestrator import Orchestrator
from core.types import Message
from providers.ollama_provider import OllamaProvider

app = FastAPI(title="GenAI API (Híbrido)")


provider = OllamaProvider(model="llama3.2:3b")
orch = Orchestrator(provider)

class ChatRequest(BaseModel):
    messages: list[dict] = []
    user_text: str

@app.post("/chat")
def chat(req: ChatRequest):
    msgs = [Message(role=m["role"], content=m["content"]) for m in req.messages]
    result = orch.run(msgs, req.user_text)
    return {
        "answer": result.answer,
        "metrics": result.metrics,
        "trace": [{"event": e.event, **e.data} for e in result.trace],
    }