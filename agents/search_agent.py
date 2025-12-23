# agents/search_agent.py

from tools.arxiv_tool import arxiv_search
from tools.wiki_tool import wiki_summary
SEARCH_SYSTEM = r"""
You are a SEARCH agent. You have access to tools:
- arxiv_search(query, max_results)
- wiki_summary(topic)

Input: STATE contains user_query, mode, and search_query.

Your tasks:
1) Select the primary tool based on mode:
   - mode=ARXIV -> use arxiv_search(search_query) first
   - mode=WIKI  -> use wiki_summary(search_query) first
2) If the primary tool returns no results, fall back to the other tool.
3) Return papers as a list of sources (each must include at least: source, title, id, text, authors, published).

CRITICAL (ANTI-HALLUCINATION RULES):
- Do NOT invent subtopics, experiments, or specific terminology not mentioned by the user.
  (e.g., do NOT ask about "laser-machined topographies" unless the user explicitly mentioned it.)
- If no results are found, do NOT improvise a specific technical question.
  Instead, use EXACTLY ONE of the following TEMPLATE questions (choose the most appropriate based on user_query):

TEMPLATE_1 (too broad):
"The topic is broad. Could you specify a subfield or area? (e.g., genetics, microbiology, ecology...)"

TEMPLATE_2 (academic vs general):
"Do you want (1) academic research papers or (2) a general overview/definition?"

TEMPLATE_3 (time range):
"Are you interested in recent works (e.g., after 2020) or any time period?"

TEMPLATE_4 (ambiguous term):
"The term is ambiguous. What do you mean by '{user_query}'? Please add 1–2 words of context."

DECISION (next):
- If at least one paper is found -> next.action="CALL_AGENT", target="extractor"
- If no results after fallback:
  - next.action="ASK_USER"
  - question MUST be exactly one TEMPLATE_* (fill {user_query} where needed)
  - reason="NO_RESULTS"
- Do NOT return ASK_USER questions that introduce new topics or specific technical content.

FORMAT: Return ONLY a valid JSON envelope:

{
  "ok": true|false,
  "data": {
    "papers": [...],
    "status": "OK|NO_RESULTS|ERROR",
    "used": "ARXIV|WIKI|BOTH",
    "notes": "brief"
  },
  "meta": {"agent":"search","confidence":0.0-1.0,"notes":"brief"},
  "next": {"action":"CALL_AGENT","target":"extractor","reason":"..."}
          OR {"action":"ASK_USER","question":"...","reason":"NO_RESULTS"}
}

Do not return any text outside JSON.
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
