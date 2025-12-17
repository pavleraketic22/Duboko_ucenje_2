import json

def safe_json_loads(text: str):
    """
    Pokušaj da izvučeš JSON iz LLM outputa.
    Vrlo jednostavno: tražimo prvi '{' i poslednji '}'.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start:end+1]
    try:
        return json.loads(chunk)
    except Exception:
        return None