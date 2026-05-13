import os

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3-70b-8192").strip()  # Default to Llama-3-70B
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "mixtral-8x7b-32768").strip()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class GroqError(Exception):
    pass

def call_groq_llm(prompt, model=None):
    if not GROQ_API_KEY:
        raise GroqError("Missing GROQ_API_KEY. Set it in .env or environment variables.")
    model = model or GROQ_MODEL
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=data)
    if resp.status_code != 200:
        raise GroqError(f"Groq API error: {resp.status_code} {resp.text}")
    result = resp.json()
    return result["choices"][0]["message"]["content"]
