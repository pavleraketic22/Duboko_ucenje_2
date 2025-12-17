import requests
from typing import Dict

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

def wiki_summary(topic: str) -> Dict:
    try:
        url = WIKI_API + topic.replace(" ", "%20")
        r = requests.get(url, timeout=20, headers={"User-Agent": "agent-system/1.0"})
        if r.status_code == 404:
            return {"ok": False, "status": "NO_RESULTS", "data": None}
        r.raise_for_status()
        data = r.json()
        extract = data.get("extract", "")
        title = data.get("title", topic)
        if not extract:
            return {"ok": False, "status": "NO_RESULTS", "data": None}
        return {
            "ok": True,
            "status": "OK",
            "data": {
                "source": "wikipedia",
                "id": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "title": title,
                "text": extract
            }
        }
    except Exception as e:
        return {"ok": False, "status": "ERROR", "error": str(e), "data": None}