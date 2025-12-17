import requests
from typing import Optional

class OllamaLLM:
    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def generate(self, prompt: str, stream: bool = False, temperature: float = 0.2, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")