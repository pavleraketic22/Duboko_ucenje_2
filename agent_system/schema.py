# agent_system/schema.py
from typing import Dict, Any, Optional

def envelope(agent: str, ok: bool, data: Dict[str, Any], confidence: float = 0.5, notes: str = "", next: Optional[Dict[str, Any]] = None):
    if next is None:
        next = {"action": "DONE", "final": ""}
    return {
        "ok": ok,
        "data": data,
        "meta": {"agent": agent, "confidence": confidence, "notes": notes},
        "next": next
    }
