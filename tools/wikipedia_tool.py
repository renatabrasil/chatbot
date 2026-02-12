import requests

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