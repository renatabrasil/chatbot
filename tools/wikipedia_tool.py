import requests

# ------------------ Tool: Wikipedia (MediaWiki API) ------------------
def wikipedia_search(query: str, limit: int = 3, lang: str = "eng") -> str:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    headers = {
        # Wikipedia/MediaWiki costuma bloquear requests sem User-Agent claro
        "User-Agent": "renata-genai-chatbot/0.1 (learning project; contact: none)",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }
    r = requests.get(url=url, params=params, headers=headers, timeout=15)
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