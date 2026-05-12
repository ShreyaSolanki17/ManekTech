import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest").strip()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma").strip()

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY. Set it in .env or environment variables.")
