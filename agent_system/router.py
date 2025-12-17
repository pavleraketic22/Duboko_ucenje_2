from agents.search_agent import SearchAgent
from agents.extractor_agent import ExtractorAgent
from agents.writer_agent import WriterAgent

class Router:
    def __init__(self, search: SearchAgent, extractor: ExtractorAgent, writer: WriterAgent):
        self.search = search
        self.extractor = extractor
        self.writer = writer

    def run(self, user_query: str):
        state = {
            "query": user_query,
            "papers": [],
            "extractions": [],
        }

        # dinamička petlja: može da se vraća nazad
        for step in range(3):  # safety cap
            # 1) ako nemamo izvore -> search
            if not state["papers"]:
                s = self.search.run(state["query"], k=2)
                if not s["ok"]:
                    # Interakcija sa korisnikom (traženo u projektu)
                    reason = s["next"]["reason"]
                    print(f"\n[Router] Nema rezultata. {reason}")
                    y = input("Dodaj preciziranje (npr. 'computer vision 2020-2024') ili Enter da prekineš: ").strip()
                    if not y:
                        return {"ok": False, "text": "Prekid: nema dovoljno informacija.", "trace": s}
                    state["query"] = state["query"] + " " + y
                    continue

                state["papers"] = s["data"]["papers"]
                continue

            # 2) ako nemamo ekstrakcije -> extractor
            if not state["extractions"]:
                e = self.extractor.run(state["papers"])
                if not e["ok"]:
                    # fallback: nova pretraga
                    state["papers"] = []
                    state["extractions"] = []
                    continue

                state["extractions"] = e["data"]["extractions"]

                # ako je premalo dobrih ekstrakcija, vrati na search
                if len(state["extractions"]) < 2:
                    state["papers"] = []
                    state["extractions"] = []
                continue

            # 3) writer
            w = self.writer.run(state["query"], state["extractions"])
            if w["ok"]:
                return {"ok": True, "text": w["data"]["text"], "sources": w["data"]["sources"], "state": state}

            # writer traži još izvora
            state["papers"] = []
            state["extractions"] = []

        return {"ok": False, "text": "Prekid: previše iteracija (loop cap).", "state": state}