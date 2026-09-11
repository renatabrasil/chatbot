from pathlib import Path
from typing import List

from certifi import contents

KB_DIR = Path(__file__).resolve().parents[1] / "kb"


def _split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def search_local_kb(query: str, max_results: int = 3) -> str:
    if not query or not query.strip():
        return "Query vazia."

    if not KB_DIR.exists():
        return f"Pasta KB não encontrada: {KB_DIR}"

    query_terms = [term.lower() for term in query.strip().split() if term.strip()]
    if not query_terms:
        return "Query sem termos válidos."

    candidates = []

    for file_path in KB_DIR.glob("*"):
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            continue

        chunks = _split_into_chunks(content, chunk_size=500)

        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for term in query_terms if term in chunk_lower)

            if score > 0:
                candidates.append(
                    {
                        "file": file_path.name,
                        "chunk_index": idx,
                        "score": score,
                        "text": chunk.strip()
                    }
                )
    if not candidates:
        return "Nenhum resultado encontrado na base local."

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:max_results]

    lines = []
    for item in top:
        lines.append(
            f"[arquivo={item['file']} trecho={item['chunk_index']} score={item['score']}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(lines)