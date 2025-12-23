# agent_system/validate.py
import json

def safe_json_loads(text: str):
    if not text:
        return None
    # pokušaj direktno
    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: nađi najveći json blok
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start:end+1]
    try:
        return json.loads(chunk)
    except Exception:
        return None
