import requests
import sys

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

def chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    print("Chatbot local (Ollama). Digite /sair para encerrar.\n")
    messages = [
        {
            "role": "system",
            "content": "Você é um assistente útil e direto."
        }
    ]

    while True:
        user = input("Você: ").strip()
        if not user:
            continue
        if user.lower() in {"/sair", "/exit", "sair"}:
            print("Até mais!")
            return

        messages.append({"role": "user", "content": user})
        try:
            answer = chat(messages)
        except Exception as e:
            print(f"[Erro] {e}")
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    main()