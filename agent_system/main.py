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
    q = input("Unesi temu/pitanje: ").strip()
    res = router.run(q)

    print("\n=== REZULTAT ===\n")
    print(res["text"])

    if res.get("sources"):
        print("\n=== IZVORI (naslovi) ===")
        for s in res["sources"]:
            print("-", s)

if __name__ == "__main__":
    main()

