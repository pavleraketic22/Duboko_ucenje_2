from agent_system.llm_ollama import OllamaLLM
from agent_system.schema import msg
from tools.arxiv_tool import arxiv_search
from tools.wiki_tool import wiki_summary

class SearchAgent:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def run(self, user_query: str, k: int = 5):
        # 1) arXiv search
        ar = arxiv_search(user_query, max_results=k)

        # 2) Wikipedia fallback (ako arXiv nema)
        wiki = None
        if not ar["ok"]:
            wiki = wiki_summary(user_query)

        papers = []
        if ar["ok"]:
            papers.extend(ar["data"])
        elif wiki and wiki["ok"]:
            papers.append(wiki["data"])

        if not papers:
            return msg(
                "SEARCH_RESULT",
                False,
                {"papers": [], "status": "NO_RESULTS"},
                agent="search",
                confidence=0.3,
                notes="No sources found",
                next={"action": "ASK_USER", "target": "router", "reason": "Preciziraj temu ili dodaj ključne reči (oblast/godine)."}
            )

        return msg(
            "SEARCH_RESULT",
            True,
            {"papers": papers, "status": "OK"},
            agent="search",
            confidence=0.7,
            notes=f"Found {len(papers)} sources",
            next={"action": "HANDOFF", "target": "extractor", "reason": "Sources collected, need extraction."}
        )