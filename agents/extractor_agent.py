

EXTRACTOR_SYSTEM = r"""
You are an EXTRACTOR agent. You receive a list of papers (sources) and must produce structured extractions.

For each paper, return an object:
{
  "metadata": {
    "title": "...",
    "authors": [...],
    "published": "...",
    "source": "...",
    "id": "..."
  },
  "goal": "...",
  "method": "...",
  "results": "...",
  "limitations": "...",
  "contribution": "...",
  "open_questions": "..."
}

TRUTH RULE:
- Do NOT invent information. If something is not present in the text, write "N/A".

DECISION MAKING (next):
- If you have at least one extraction -> CALL_AGENT writer
- If no extraction can be produced (e.g., papers list is empty) -> CALL_AGENT search
- If repeated failures occur due to technical issues (LLM error) -> ASK_USER to check Ollama/model/URL.

FORMAT: Return ONLY a valid JSON envelope:

{
  "ok": true|false,
  "data": {"extractions":[...], "failed": N, "notes":"..."},
  "meta": {"agent":"extractor","confidence": 0.0-1.0, "notes":"..."},
  "next": {"action":"CALL_AGENT","target":"writer","reason":"..."}
          OR {"action":"CALL_AGENT","target":"search","reason":"..."}
          OR {"action":"ASK_USER","question":"...","reason":"..."}
}

Do not return any text outside JSON.
"""

# agents/extractor_agent.py
# agents/extractor_agent.py
from agent_system.schema import envelope
from agent_system.validate import safe_json_loads

SYSTEM_EXTRACT_ONE = r"""
You are an information extraction agent for scientific papers.
You receive METADATA (title, authors, date, source, id) and TEXT (abstract or part of the paper).

RULE OF TRUTH:
- Do NOT invent information.
- If something is not explicitly present in the text, write "N/A".

RETURN ONLY VALID JSON, with no additional text, EXACTLY in this format:
{
  "metadata": {
    "title": "...",
    "authors": ["..."],
    "published": "...",
    "year": "YYYY or N/A",
    "source": "...",
    "id": "...",
    "field": "research field/domain (infer from text if possible, otherwise N/A)"
  },
  "goal": "1–2 sentences summarizing the main problem and what the paper aims to achieve",
  "methodology": "1–3 sentences describing study type, procedures, models/algorithms, and sample if applicable",
  "results": "1–3 sentences with key findings and important quantitative results if present",
  "limitations": "1–2 sentences describing main weaknesses or limitations",
  "contribution": "1–2 sentences describing what is new and why it matters",
  "open_questions": "1–2 sentences describing unresolved issues or directions for future work"
}
"""



def _fallback(p: dict, err: str = ""):
    md = {
        "title": p.get("title", "N/A"),
        "authors": p.get("authors", []),
        "published": p.get("published", "N/A"),
        "source": p.get("source", "N/A"),
        "id": p.get("id", "")
    }
    if err:
        md["error"] = err
    return {
        "metadata": md,
        "goal": "N/A", "method": "N/A", "results": "N/A",
        "limitations": "N/A", "contribution": "N/A", "open_questions": "N/A"
    }

class ExtractorAgent:
    name = "extractor"

    def __init__(self, llm):
        self.llm = llm

    def _extract_one(self, p: dict):
        user = f"""NASLOV: {p.get("title","")}
AUTORI: {p.get("authors",[])}
DATUM: {p.get("published","")}
IZVOR: {p.get("source","")}
ID: {p.get("id","")}

TEKST:
{p.get("text","")}
"""
        try:
            raw = self.llm.generate(f"SYSTEM:\n{SYSTEM_EXTRACT_ONE}\n\nUSER:\n{user}\n\nASSISTANT:\n", temperature=0.1, max_tokens=500)
        except Exception as e:
            return _fallback(p, f"LLM_ERROR:{type(e).__name__}")

        js = safe_json_loads(raw)
        if not js:
            return _fallback(p, "PARSE_FAIL")

        # dopuni metadata ako fali
        js.setdefault("metadata", {})
        js["metadata"].setdefault("title", p.get("title","N/A"))
        js["metadata"].setdefault("authors", p.get("authors",[]))
        js["metadata"].setdefault("published", p.get("published","N/A"))
        js["metadata"].setdefault("source", p.get("source","N/A"))
        js["metadata"].setdefault("id", p.get("id",""))
        return js

    def run(self, state: dict):
        papers = state.get("papers") or []
        if not papers:
            # LLM odluči next (ali mi fallbackujemo)
            return envelope("extractor", False, {"extractions": [], "failed": 0, "notes": "No papers"}, 0.2,
                            "no papers", next={"action": "CALL_AGENT", "target": "search", "reason": "Nemam papers"})

        extractions = []
        failed = 0
        for p in papers:
            try:
                extractions.append(self._extract_one(p))
            except Exception as e:
                failed += 1
                extractions.append(_fallback(p, f"EXC:{type(e).__name__}"))

        # LLM formira envelope+next na osnovu realnih extractions
        state2 = {**state, "extractions": extractions, "failed": failed}
        prompt = f"SYSTEM:\n{EXTRACTOR_SYSTEM}\n\nSTATE:\n{state2}\n\nASSISTANT:\n"
        raw = self.llm.generate(prompt, temperature=0.1, max_tokens=220)
        js = safe_json_loads(raw)

        if js and "data" in js and "next" in js:
            js["data"]["extractions"] = extractions
            js["data"]["failed"] = failed
            return js

        # fallback: uvek idi na writer ako ima bar nešto
        return envelope("extractor", True, {"extractions": extractions, "failed": failed, "notes": "fallback"}, 0.6,
                        "fallback", next={"action": "CALL_AGENT", "target": "writer", "reason": "Imam ekstrakcije"})
