WRITER_SYSTEM = r"""
Ti si WRITER agent. Dobijaš STATE koji sadrži: user_query i extractions (lista ekstrakcija).
Koristi ISKLJUČIVO informacije iz extractions. Ne izmišljaj. Ako nema -> napiši "N/A".

U data.final_text i next.final VRATI SAMO sledeći izveštaj (bez ikakvih dodatnih pravila, bez objašnjenja, bez naslova van šablona):

Rutina za analiziranje naučnog rada:
Ulaz: tekst rada
Izlaz: kratak strukturisan izvestaj

METAPODACI
Naslov: ...
Autori: ...
Godina: ...
Oblast: ...
Izvor/Link: ...

CILJ RADA
...

METODOLOGIJA
...

REZULTATI
...

OGRANICENJA
...

DOPRINOS
...

OTVORENA PITANJA
...

JSON pravila:
- Vrati ISKLJUČIVO validan JSON envelope (bez teksta pre/posle).
- next.action mora biti "DONE".
- next.final mora biti NEPRAZAN string.
- data.final_text mora biti IDENTICAN next.final (isti sadržaj).
- U final_text/final NE SMEŠ dodati ništa osim gornjeg izveštaja.

VRATI JSON tačno ovog oblika (popuni vrednosti, ne dodaj nova polja):
{
  "ok": true,
  "data": {"final_text": "", "sources": [], "notes": ""},
  "meta": {"agent": "writer", "confidence": 0.0, "notes": ""},
  "next": {"action": "DONE", "final": ""}
}
"""


# agents/writer_agent.py
from agent_system.validate import safe_json_loads
from agent_system.schema import envelope

class WriterAgent:
    name = "writer"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: dict):
        prompt = f"SYSTEM:\n{WRITER_SYSTEM}\n\nSTATE:\n{state}\n\nASSISTANT:\n"
        raw = self.llm.generate(prompt, temperature=0.2, max_tokens=900)
        js = safe_json_loads(raw)

        # 1) Ako je LLM vratio envelope, ispravi ga
        if isinstance(js, dict) and "next" in js:
            data = js.get("data") or {}
            nxt = js.get("next") or {}

            # uzmi final_text ako postoji
            final_text = (data.get("final_text") or "").strip()

            # 1a) Ako je next.final slučajno JSON string, probaj da ga razložiš
            nxt_final = nxt.get("final")
            if isinstance(nxt_final, str) and nxt_final.strip().startswith("{"):
                nested = safe_json_loads(nxt_final)
                if isinstance(nested, dict):
                    nested_text = (((nested.get("data") or {}).get("final_text")) or "").strip()
                    if nested_text:
                        final_text = nested_text
                        js.setdefault("data", {})["final_text"] = final_text

            # 1b) Ako je akcija DONE, final MORA biti plain text
            if nxt.get("action") == "DONE":
                if not final_text:
                    # ako LLM nije dao final_text, fallback na raw (ali raw nije JSON)
                    final_text = raw.strip()
                    js.setdefault("data", {})["final_text"] = final_text
                js["next"]["final"] = final_text

            return js

        # 2) fallback: LLM nije vratio envelope -> DONE sa sirovim tekstom
        return envelope(
            "writer",
            True,
            {"final_text": raw, "sources": state.get("sources", [])},
            confidence=0.6,
            notes="fallback",
            next={"action": "DONE", "final": raw}
        )

