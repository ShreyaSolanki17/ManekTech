# Legal Document Intelligence (CAAD Tax Decisions)

This project scrapes Portuguese tax law decisions from CAAD, extracts structured JSON with an LLM, builds a simple vector database, and serves a RAG Q&A UI.

## Setup

1) Create and activate a virtual environment
- Windows PowerShell:
	- `python -m venv .venv`
	- `./.venv/Scripts/Activate.ps1`

2) Install dependencies
- `pip install -r requirements.txt`

3) Configure environment
- Copy `.env.example` to `.env` and set `GEMINI_API_KEY`

## Run the pipeline

**Note:**
For demonstration and testing, we have chosen to process 50 cases (`--limit 50`). This number is sufficient to show the pipeline works end-to-end, is manageable for local resources, and meets the practical requirements of the task. You can increase this limit for larger experiments if desired.

1) Scrape CAAD decisions
- `python src/scrape_caad.py --max-pages 3 --limit 50`

2) Extract structured JSON with LLM
- `python src/extract_cases.py --input data/raw/raw_index.jsonl --output data/processed/cases.jsonl`

3) Build vector database
- `python src/build_vector_db.py --input data/processed/cases.jsonl --persist data/chroma`

4) Start the UI
- `streamlit run src/ui_app.py`

## Notes
- No API keys are committed. Use `.env` locally only.
- You can change models via environment variables:
	- `GEMINI_MODEL` (default: `gemini-1.5-flash-latest`)

## Project structure

- `src/scrape_caad.py` - scrape listing pages and case text
- `src/extract_cases.py` - LLM extraction into strict JSON (Gemini)
- `src/build_vector_db.py` - embed and store in Chroma (Gemini)
- `src/rag_qa.py` - retrieval and answer generation (Gemini)
- `src/ui_app.py` - Streamlit UI
- `src/gemini_client.py` - Gemini API client
- `data/raw` - scraped raw text
- `data/processed` - extracted JSON
- `data/chroma` - vector DB
