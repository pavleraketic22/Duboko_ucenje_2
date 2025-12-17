from typing import Dict, Any

def msg(type: str, ok: bool, data: Dict[str, Any], agent: str, confidence: float = 0.5, notes: str = "", next=None):
    if next is None:
        next = {"action": "HANDOFF", "target": "router", "reason": ""}
    return {
        "type": type,
        "ok": ok,
        "data": data,
        "meta": {
            "agent": agent,
            "confidence": confidence,
            "notes": notes,
        },
        "next": next
    }
