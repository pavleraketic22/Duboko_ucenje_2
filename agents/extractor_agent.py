from llm_ollama import OllamaLLM
from schema import msg
from validate import safe_json_loads

SYSTEM = """Ti si analitički agent za naučne izvore.
Iz ulaznog TEKSTA izvuci strukturisane informacije.

VRATI ISKLJUČIVO VALIDAN JSON sa poljima:
{
  "metadata": {"title": "...", "authors": [...], "published": "...", "source": "..."},
  "goal": "...",
  "method": "...",
  "results": "...",
  "limitations": "...",
  "contribution": "...",
  "open_questions": "..."
}

Ne izmišljaj: ako nema u tekstu, napiši "N/A".
"""

class ExtractorAgent:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def _extract_one(self, paper: dict):
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        published = paper.get("published", "")
        source = paper.get("source", "")
        text = paper.get("text", "")

        user = f"""NASLOV: {title}
AUTORI: {authors}
DATUM: {published}
IZVOR: {source}

TEKST:
{text}
"""
        raw = self.llm.generate(f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{user}\n\nASSISTANT:\n", temperature=0.1, max_tokens=500)
        js = safe_json_loads(raw)
        if js is None:
            # fallback: minimalna ekstrakcija bez rušenja
            js = {
                "metadata": {"title": title, "authors": authors, "published": published, "source": source},
                "goal": "N/A", "method": "N/A", "results": "N/A",
                "limitations": "N/A", "contribution": "N/A", "open_questions": "N/A"
            }
        return js

    def run(self, papers: list):
        extractions = []
        failed = 0
        for p in papers:
            try:
                extractions.append(self._extract_one(p))
            except Exception:
                failed += 1

        if len(extractions) == 0:
            return msg(
                "EXTRACT_RESULT",
                False,
                {"extractions": [], "failed": failed},
                agent="extractor",
                confidence=0.2,
                notes="Extraction failed for all sources",
                next={"action": "HANDOFF", "target": "search", "reason": "Try new sources / new query."}
            )

        return msg(
            "EXTRACT_RESULT",
            True,
            {"extractions": extractions, "failed": failed},
            agent="extractor",
            confidence=0.7,
            notes=f"Extracted {len(extractions)} items (failed={failed})",
            next={"action": "HANDOFF", "target": "writer", "reason": "Need synthesis."}
        )