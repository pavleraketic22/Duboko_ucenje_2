from agent_system.llm_ollama import OllamaLLM
from agent_system.schema import msg
from agent_system.validate import safe_json_loads

SYSTEM = """Ti si Query Rewriter za arXiv/Wikipedia pretragu.
Zadatak: iz korisničke rečenice napravi kratak SEARCH_QUERY.

PRAVILA:
- Vrati ISKLJUČIVO validan JSON
- Polje "search_query" mora biti string od 3 do 10 ključnih reči
- Ne vraćaj celu rečenicu, ne vraćaj objašnjenja
- Ako je tema opšta (ličnost, praznik, pojam iz kulture), napiši "mode": "WIKI"
- Ako je tema istraživačka/ML/CS, napiši "mode": "ARXIV"
- Ako ne možeš, vrati "mode": "ASK_USER" i "search_query": ""

FORMAT:
{"mode":"ARXIV|WIKI|ASK_USER","search_query":"...","notes":"..."}
"""

class QueryAgent:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def run(self, user_query: str):
        prompt = f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{user_query}\n\nASSISTANT:\n"
        raw = self.llm.generate(prompt, temperature=0.1, max_tokens=120)

        js = safe_json_loads(raw)
        if not js or "mode" not in js:
            # fallback: ako je LLM vratio haos
            js = {"mode": "ARXIV", "search_query": user_query, "notes": "fallback"}

        return msg(
            "QUERY_RESULT",
            True,
            {"mode": js.get("mode", "ARXIV"), "search_query": js.get("search_query", user_query), "notes": js.get("notes", "")},
            agent="query",
            confidence=0.7,
            next={"action": "HANDOFF", "target": "search", "reason": "Prepared search query."}
        )
