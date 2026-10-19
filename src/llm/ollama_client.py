import os

import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")  # change the model in .env, not in code

def ask_ollama(prompt: str, temperature: float | None = None, model: str | None = None) -> str:
    payload = {"model": model or DEFAULT_MODEL, "prompt": prompt, "stream": False, "think": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}  # 0 = same answer every time
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()       # turns HTTP 404/500 into a clear Python error
    return response.json()["response"]  # Ollama puts the answer under "response"
