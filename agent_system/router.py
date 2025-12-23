from agents.search_agent import SearchAgent
from agents.extractor_agent import ExtractorAgent
from agents.writer_agent import WriterAgent
from cache.rag import SourceRAG

class Router:
    def __init__(self, search: SearchAgent, extractor: ExtractorAgent, writer: WriterAgent):
        self.search = search
        self.extractor = extractor
        self.writer = writer

    def run(self, user_query: str):
        rag = SourceRAG()
        state = {"query": user_query, "papers": [], "extractions": []}

        for step in range(12):

            if not state["papers"]:
                hit = rag.get(state["query"], threshold=0.8)
                if hit:
                    print(f"[Router] RAG HIT (score={hit['score']:.2f}) -> preskačem Search")
                    state["papers"] = hit["papers"]


            if not state["papers"]:
                print("------------PRETRAZUJEM--------------")
                s = self.search.run(state["query"], k=2)
                if not s["ok"]:
                    reason = s["next"]["reason"]
                    print(f"\n[Router] Nema rezultata. {reason}")
                    y = input("Dodaj preciziranje ili Enter da prekineš: ").strip()
                    if not y:
                        return {"ok": False, "text": "Prekid: nema dovoljno informacija.", "trace": s}
                    state["query"] = state["query"] + " " + y
                    continue

                state["papers"] = s["data"]["papers"]
                rag.add(state["query"], state["papers"])
                continue


            if not state["extractions"]:
                print("-------------EKSTRAKTUJEM------------------")
                e = self.extractor.run(state["papers"])
                if not e["ok"]:
                    state["papers"] = []
                    state["extractions"] = []
                    continue

                state["extractions"] = e["data"]["extractions"]
                continue


            print("------------PISEM----------------")
            w = self.writer.run(state["query"], state["extractions"])
            if w["ok"]:
                return {"ok": True, "text": w["data"]["text"], "sources": w["data"]["sources"], "state": state}


            state["papers"] = []
            state["extractions"] = []

        return {"ok": False, "text": "Prekid: previše iteracija (loop cap).", "state": state}
