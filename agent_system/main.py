from llm_ollama import OllamaLLM
from agents.search_agent import SearchAgent
from agents.extractor_agent import ExtractorAgent
from agents.writer_agent import WriterAgent
from router import Router

def main():
    llm = OllamaLLM(model="llama3.2")

    search = SearchAgent(llm)
    extractor = ExtractorAgent(llm)
    writer = WriterAgent(llm)

    router = Router(search, extractor, writer)

    print("=== Multi-Agent Research Assistant (local Ollama) ===")

    while True:
        q = input("\nUnesi temu/pitanje (ili 'exit' za kraj): ").strip()
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("Pozdrav!")
            break

        res = router.run(q)

        print("\n=== REZULTAT ===\n")
        print(res.get("text", "Nema teksta."))

        if res.get("sources"):
            print("\n=== IZVORI (naslovi) ===")
            for s in res["sources"]:
                print("-", s)

        action = input("\nŠta dalje? (Enter = novo pitanje, 'exit' = kraj): ").strip().lower()
        if action in {"exit", "quit", "q"}:
            print("Pozdrav!")
            break

if __name__ == "__main__":
    main()
