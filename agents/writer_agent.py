from llm_ollama import OllamaLLM
from schema import msg

SYSTEM = """Ti si agent za sintezu teksta za naučno istraživanje.
KORISTI SAMO informacije iz ulaznih ekstrakcija.
Ako nešto nije prisutno u ekstrakcijama, jasno reci da nije pronađeno.
Ne izmišljaj citate ili rezultate.

Napiši strukturisan tekst:
1) Kratak pregled teme
2) Ključni pristupi/metode
3) Najvažniji nalazi (ako postoje)
4) Ograničenja i otvorena pitanja
5) Lista izvora (naslovi)
"""

class WriterAgent:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def run(self, user_query: str, extractions: list):
        user = f"TEMA: {user_query}\n\nEKSTRAKCIJE:\n{extractions}\n"
        text = self.llm.generate(f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{user}\n\nASSISTANT:\n", temperature=0.2, max_tokens=900)

        # ako je baš prazno, traži još izvora
        if not text or len(text.strip()) < 30:
            return msg(
                "WRITE_RESULT",
                False,
                {"text": "", "sources": []},
                agent="writer",
                confidence=0.2,
                notes="Empty synthesis",
                next={"action": "HANDOFF", "target": "search", "reason": "Need more/better sources."}
            )

        titles = []
        for e in extractions:
            md = e.get("metadata", {})
            t = md.get("title")
            if t:
                titles.append(t)

        return msg(
            "WRITE_RESULT",
            True,
            {"text": text, "sources": titles},
            agent="writer",
            confidence=0.75,
            notes="Synthesis complete",
            next={"action": "DONE", "target": "router", "reason": "Answer ready."}
        )
