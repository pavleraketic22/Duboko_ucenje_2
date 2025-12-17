import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

ARXIV_API = "http://export.arxiv.org/api/query"

def arxiv_search(query: str, max_results: int = 5) -> Dict:
    """
    Vraća listu radova: title, id, authors, summary, published
    """
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        }
        r = requests.get(ARXIV_API, params=params, timeout=30)
        r.raise_for_status()

        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}

        entries = []
        for entry in root.findall("a:entry", ns):
            title = (entry.find("a:title", ns).text or "").strip()
            arxiv_id = (entry.find("a:id", ns).text or "").strip()
            summary = (entry.find("a:summary", ns).text or "").strip()
            published = (entry.find("a:published", ns).text or "").strip()
            authors = [a.find("a:name", ns).text.strip() for a in entry.findall("a:author", ns)]

            entries.append({
                "source": "arxiv",
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "text": summary  # za extractor (minimum: abstract)
            })

        if not entries:
            return {"ok": False, "status": "NO_RESULTS", "data": []}

        return {"ok": True, "status": "OK", "data": entries}

    except Exception as e:
        return {"ok": False, "status": "ERROR", "error": str(e), "data": []}

