# main.py
from agent_system.llm_ollama import OllamaLLM
from agent_system.runner import Orchestrator
from agents.query_agent import QueryAgent
from agents.search_agent import SearchAgent
from agents.extractor_agent import ExtractorAgent
from agents.writer_agent import WriterAgent

def main():
    llm = OllamaLLM(model="llama3.2")

    agents = {
        "query": QueryAgent(llm),
        "search": SearchAgent(llm),
        "extractor": ExtractorAgent(llm),
        "writer": WriterAgent(llm),
    }

    orch = Orchestrator(agents, max_steps=20)

    print("=== Multi-Agent Research Assistant (dynamic handoff, local Ollama) ===")

    while True:
        q = input("\nUnesi temu/pitanje (ili 'exit' za kraj): ").strip()
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("Pozdrav!")
            break

        res = orch.run(q)

        print("\n=== REZULTAT ===\n")
        print(res.get("text", "Nema teksta."))

        if res.get("sources"):
            print("\n=== IZVORI (naslovi) ===")
            for s in res["sources"]:
                print("-", s)

if __name__ == "__main__":
    main()
