from agent_system.llm_ollama import OllamaLLM

from agent_system.validate import safe_json_loads

QUERY_SYSTEM = r"""
You are a QUERY agent in a multi-agent system.

Input: you receive STATE (JSON) containing user_query and possibly previous results.

Your tasks:
1) Normalize user_query into a search_query (3–10 keywords, preferably in English).
2) Choose mode:
   - "ARXIV" for academic / ML / CS topics,
   - "WIKI" for general concepts or definitions.
3) Optionally set constraints (domain, years) if you can infer them.

IMPORTANT: You must also decide the next step (next).
- If user_query is vague (e.g., "rag", "agent"):
  - Do NOT ask the user immediately if you can reasonably infer the most common meaning.
  - If ambiguity is critical (e.g., "rag" could mean multiple things), return ASK_USER with explicit options.
- In normal cases, after success -> CALL_AGENT search.
- If you cannot determine a reasonable interpretation -> ASK_USER.

FORMAT: Return ONLY a valid JSON envelope:

{
  "ok": true|false,
  "data": {
    "mode": "ARXIV|WIKI|ASK_USER",
    "search_query": "...",
    "constraints": {"domain":"...", "years":"..."},
    "notes": "..."
  },
  "meta": {"agent":"query","confidence":0.0-1.0,"notes":"..."},
  "next":
    {"action":"CALL_AGENT","target":"search","reason":"..."}
    OR {"action":"ASK_USER","question":"...","reason":"..."}
}

Do not return any text outside JSON.
"""


# agents/query_agent.py
from agent_system.validate import safe_json_loads
from agent_system.schema import envelope

class QueryAgent:
    name = "query"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: dict):
        user_query = state.get("user_query", "")

        prompt = f"SYSTEM:\n{QUERY_SYSTEM}\n\nSTATE:\n{state}\n\nASSISTANT:\n"
        raw = self.llm.generate(prompt, temperature=0.1, max_tokens=300)
        js = safe_json_loads(raw)

        if not js or "data" not in js or "next" not in js:
            # fallback (bez LLM odluke)
            return envelope(
                "query", True,
                {"mode": "ARXIV", "search_query": user_query, "constraints": {}, "notes": "fallback"},
                confidence=0.4,
                notes="LLM parse fail",
                next={"action": "CALL_AGENT", "target": "search", "reason": "fallback"}
            )

        return js
