from utils_chunking import chunk_text
# Add missing import for os
import os
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from gemini_client import get_gemini_model
from groq_client import call_groq_llm, GROQ_MODEL, GROQ_MODEL_FALLBACK, GroqError
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from config import GEMINI_MODEL
from utils_json import safe_json_loads

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extract_cases_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8").strip()


class CaseSchema(BaseModel):
    case_id: str
    decision_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    plaintiff_name: str
    defendant_name: str
    court_decision: str
    summary: str
    language: str = Field(pattern=r"^Portuguese$")


def call_llm_with_fallback(case_id: str, raw_text: str) -> Dict[str, Any]:
    """
    Try Gemini first, then Groq Llama-3, then Groq Mixtral if needed.
    """
    # If text is too large, chunk and summarize before extraction
    MAX_TOKENS = 6000  # Conservative for Groq
    SAFE_CHARS = 12000
    if len(raw_text) > SAFE_CHARS:
        print(f"Case {case_id} too large, chunking and summarizing...")
        chunks = chunk_text(raw_text, max_tokens=MAX_TOKENS - 500, overlap=200)
        summarized_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"Resuma o seguinte texto jurídico em português, mantendo todos os detalhes essenciais para extração de informações legais.\n\nTexto ({i+1}/{len(chunks)}):\n" + chunk
            # Try Gemini for summarization
            try:
                model = get_gemini_model()
                response = model.generate_content(chunk_prompt)
                summarized_chunks.append(response.text.strip())
            except Exception as exc:
                print(f"Summarization failed for chunk {i+1} of case {case_id}: {exc}")
                summarized_chunks.append(chunk[:SAFE_CHARS])  # fallback: truncate
        summarized_text = '\n'.join(summarized_chunks)
        safe_raw_text = summarized_text.replace('{', '{{').replace('}', '}}')
    else:
        safe_raw_text = raw_text[:SAFE_CHARS].replace('{', '{{').replace('}', '}}')

    prompt = PROMPT_TEMPLATE.replace("<<CASE_ID>>", case_id).replace("<<RAW_TEXT>>", safe_raw_text)
    # Try Gemini
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        return safe_json_loads(response.text)
    except Exception as gemini_exc:
        print(f"Gemini failed for {case_id}: {gemini_exc}\nTrying Groq Llama-3...")
        # Try Groq Llama-3
        try:
            groq_response = call_groq_llm(prompt, model=GROQ_MODEL)
            return safe_json_loads(groq_response)
        except Exception as groq_exc:
            print(f"Groq Llama-3 failed for {case_id}: {groq_exc}\nTrying Groq Mixtral...")
            # Try Groq Mixtral
            try:
                groq_response2 = call_groq_llm(prompt, model=GROQ_MODEL_FALLBACK)
                return safe_json_loads(groq_response2)
            except Exception as groq2_exc:
                print(f"Groq Mixtral failed for {case_id}: {groq2_exc}")
                raise RuntimeError(f"All LLMs failed for {case_id}")


def extract_cases(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for rec in tqdm(records, desc="Extracting JSON"):
        for attempt in range(2):
            try:
                data = call_llm_with_fallback(rec["case_id"], rec["raw_text"])
                validated = CaseSchema(**data)
                results.append(validated.model_dump())
                break
            except (ValidationError, json.JSONDecodeError) as exc:
                if attempt == 1:
                    print(f"Failed to parse case {rec['case_id']}: {exc}")
            except Exception as exc:
                if attempt == 1:
                    print(f"LLM error for case {rec['case_id']}: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/raw_index.jsonl")
    parser.add_argument("--output", default="data/processed/cases.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Load already extracted case_ids
    existing = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing[obj["case_id"]] = obj
                except Exception:
                    continue

    # Load all records
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["case_id"] not in existing:
                records.append(rec)

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("All cases already extracted. Nothing to do.")
        return

    extracted = extract_cases(records)

    # Append new results to output file
    if extracted:
        with open(args.output, "a", encoding="utf-8") as f:
            for item in extracted:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Extracted {len(extracted)} new cases. Total now: {len(existing) + len(extracted)}")
    else:
        print("No new cases extracted. Output file left unchanged.")


if __name__ == "__main__":
    main()
