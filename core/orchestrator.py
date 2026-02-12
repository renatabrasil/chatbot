

def detect_tool(user_text: str):
    t = user_text.strip().lower()
    if t.startswith("/wiki"):
        return "wikipedia_search", user_text[len("/wiki "):].strip()
    return None, None