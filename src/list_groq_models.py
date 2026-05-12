import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/models"

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Set it in .env or environment variables.")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
}
resp = requests.get(GROQ_API_URL, headers=headers)
if resp.status_code != 200:
    print(f"Groq API error: {resp.status_code} {resp.text}")
    exit(1)
models = resp.json()["data"]
print("Available Groq models:")
for m in models:
    print("-", m["id"])
