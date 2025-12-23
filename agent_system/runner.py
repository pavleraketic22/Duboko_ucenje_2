# runner.py
from agent_system.llm_ollama import OllamaLLM
from agents.query_agent import QueryAgent
from agents.search_agent import SearchAgent
from agents.extractor_agent import ExtractorAgent
from agents.writer_agent import WriterAgent

AGENT_ORDER = {"query", "search", "extractor", "writer"}

def run_conversation(user_query: str, agents: dict, max_steps: int = 15):
    state = {
        "user_query": user_query,
        "mode": None,
        "search_query": None,
        "constraints": {},
        "papers": [],
        "extractions": [],
        "sources": [],
        "last_agent": None,
        "last_error": None,
        "logs": []
    }

    current = "query"
    last = None

    for _ in range(max_steps):
        agent = agents[current]
        out = agent.run(state)

        state["last_agent"] = current
        state["logs"].append({"agent": current, "ok": out.get("ok"), "next": out.get("next")})

        # apply patch
        data = out.get("data", {}) or {}
        for k, v in data.items():
            state[k] = v

        nxt = (out.get("next") or {})
        action = nxt.get("action")

        print(f"[{current}] ok={out.get('ok')} next={nxt}")

        if action == "DONE":
            return {"ok": True, "text": nxt.get("final", ""), "state": state}

        if action == "ASK_USER":
            return {"ok": False, "ask_user": nxt.get("question", "Preciziraj."), "state": state}

        if action == "CALL_AGENT":
            target = nxt.get("target")
            if target not in agents:
                return {"ok": False, "text": f"Nepoznat agent: {target}", "state": state}
            current = target
            last = out
            continue

        return {"ok": False, "text": f"Nepoznata akcija: {action}", "state": state}

    return {"ok": False, "text": "Prekid: previše koraka.", "state": state}


def main():
    llm = OllamaLLM(model="llama3.2")
    agents = {
        "query": QueryAgent(llm),
        "search": SearchAgent(llm),
        "extractor": ExtractorAgent(llm),
        "writer": WriterAgent(llm),
    }
    pending_question = None
    base_query = None

    print("=== Multi-Agent (agents decide next via prompts) ===")

    while True:
        user_input = input("\nUnesi temu/pitanje (ili 'exit' za kraj): ").strip()
        if not user_input or user_input.lower() in {"exit", "quit", "q"}:
            print("Pozdrav!")
            break

        # ⬇⬇⬇ KLJUČNA LOGIKA ⬇⬇⬇
        if pending_question:
            # ovo je odgovor na prethodno ASK_USER pitanje
            q = f"{base_query} | clarification: {user_input}"
            pending_question = None
        else:
            # novo pitanje
            q = user_input
            base_query = user_input

        res = run_conversation(q, agents)

        if res.get("ok"):
            print("\n=== REZULTAT ===\n")
            print(res["text"])
        else:
            if "ask_user" in res:
                pending_question = res["ask_user"]
                print("\n[Potrebno pojašnjenje]")
                print(pending_question)
            else:
                print("\n[Greška]")
                print(res.get("text"))

if __name__ == "__main__":
    main()
