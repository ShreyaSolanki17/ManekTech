import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY. Set it in .env or environment variables.")
genai.configure(api_key=api_key)

print("Available Gemini models:")
for model in genai.list_models():
    print(f"- {model.name} (generation: {model.supported_generation_methods})")
