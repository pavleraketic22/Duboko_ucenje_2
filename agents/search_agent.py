# agents/search_agent.py

from tools.arxiv_tool import arxiv_search
from tools.wiki_tool import wiki_summary
SEARCH_SYSTEM = r"""
You are a SEARCH agent. You have access to:
- arxiv_search(query, max_results)
- wiki_summary(topic)

Input: STATE contains user_query, mode, search_query.

Your task:
1) Use the tool indicated by mode:
   - ARXIV -> arxiv_search
   - WIKI  -> wiki_summary
2) If ARXIV returns no results, you MAY fall back to WIKI.
3) Return papers as a list of sources (source, title, id, text, authors, published).

IMPORTANT DECISION LOGIC:
- Do NOT ask the user just because ARXIV returned no results.
- If WIKI returns valid content, this counts as SUCCESS.
- Ask the user ONLY if the TOPIC ITSELF is too broad or ambiguous,
  not because tools returned no results.

WHEN TO ASK THE USER (ONLY THESE CASES):
- The topic is extremely broad (e.g., "biology", "history", "AI").
- The term is ambiguous (e.g., "rag", "cell", "model").
- The user intent (academic vs general) cannot be inferred.

ALLOWED QUESTIONS (choose ONE if needed):
1) Broad topic:
"The topic is very broad. Could you specify a subfield? (e.g., genetics, microbiology, ecology)"
2) Ambiguous term:
"The term '{user_query}' is ambiguous. Please add 1–2 words of context."
3) Academic vs general:
"Do you want academic research papers or a general overview?"

DECISION (next):
- If you have ANY valid papers (ARXIV or WIKI) -> CALL_AGENT extractor
- Ask the user ONLY using one of the questions above
- reason must reflect BROAD_TOPIC or AMBIGUOUS_TERM

FORMAT: Return ONLY valid JSON:

{
  "ok": true|false,
  "data": {
    "papers": [...],
    "status": "OK|NO_RESULTS",
    "used": "ARXIV|WIKI|BOTH",
    "notes": ""
  },
  "meta": {"agent":"search","confidence":0.0-1.0,"notes":""},
  "next":
    {"action":"CALL_AGENT","target":"extractor","reason":"..."}
    OR {"action":"ASK_USER","question":"...","reason":"BROAD_TOPIC|AMBIGUOUS_TERM"}
}
"""


# agents/search_agent.py
from agent_system.schema import envelope
from agent_system.validate import safe_json_loads
from tools.arxiv_tool import arxiv_search
from tools.wiki_tool import wiki_summary

class SearchAgent:
    name = "search"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: dict):
        mode = state.get("mode") or "ARXIV"
        q = state.get("search_query") or state.get("user_query") or ""

        papers = []
        used = None

        # izvrši alatke deterministički
        if mode == "ARXIV":
            ar = arxiv_search(q, max_results=5)
            if ar["ok"]:
                papers = ar["data"]
                used = "ARXIV"
            else:
                wk = wiki_summary(q)
                if wk["ok"]:
                    papers = [wk["data"]]
                    used = "WIKI"
        else:
            wk = wiki_summary(q)
            if wk["ok"]:
                papers = [wk["data"]]
                used = "WIKI"
            else:
                ar = arxiv_search(q, max_results=5)
                if ar["ok"]:
                    papers = ar["data"]
                    used = "ARXIV"

        # sada LLM formira envelope + next (ali na osnovu realnih rezultata)
        state2 = {**state, "papers": papers, "search_used": used}
        prompt = f"SYSTEM:\n{SEARCH_SYSTEM}\n\nSTATE:\n{state2}\n\nASSISTANT:\n"
        raw = self.llm.generate(prompt, temperature=0.1, max_tokens=250)
        js = safe_json_loads(raw)

        if js and "data" in js and "next" in js:
            # obavezno ubaci stvarne papers (ne veruj LLM-u da ih izmisli)
            js["data"]["papers"] = papers
            js["data"]["used"] = used or "NONE"
            return js

        # fallback bez LLM-a
        if papers:
            return envelope("search", True, {"papers": papers, "status": "OK", "used": used}, 0.7, "fallback", next={"action": "CALL_AGENT", "target": "extractor", "reason": "Imam papers"})
        return envelope("search", False, {"papers": [], "status": "NO_RESULTS", "used": used}, 0.3, "fallback", next={"action": "ASK_USER", "question": "Nisam našao izvore. Koja oblast (npr LLM/IR) i period (npr 2020-2024)?", "reason": "NO_RESULTS"})
