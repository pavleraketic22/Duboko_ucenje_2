from agent_system.llm_ollama import OllamaLLM
from agents.search_agent import SearchAgent

def main():
    # 1) inicijalizuj LLM (isti kao u projektu)
    llm = OllamaLLM(model="llama3.2")

    # 2) inicijalizuj search agenta
    search_agent = SearchAgent(llm)

    # 3) test upit
    query = "TF-IDF text classification"

    print("=== TEST SEARCH AGENTA ===")
    print("Upit:", query)
    print("--------------------------")

    result = search_agent.run(query, k=3)

    print("OK:", result["ok"])
    print("STATUS:", result["data"].get("status"))
    print("BROJ IZVORA:", len(result["data"].get("papers", [])))
    print("\nCEO REZULTAT:\n")
    print(result)

if __name__ == "__main__":
    main()
