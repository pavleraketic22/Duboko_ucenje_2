import requests

def query_llama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(url, json=payload)
    return r.json()["response"]

print(query_llama("Kratko i jasno objasni LLM arhitekturu"))
