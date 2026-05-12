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
    safe_raw_text = raw_text[:12000].replace('{', '{{').replace('}', '}}')
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

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    extracted = extract_cases(records)

    # Write to a temp file first
    import tempfile, shutil
    temp_path = None
    if extracted:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(Path(args.output).parent)) as tf:
            temp_path = tf.name
            for item in extracted:
                tf.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Only replace output if at least one case was extracted
        shutil.move(temp_path, args.output)
        print(f"Extracted {len(extracted)} of {len(records)} cases")
    else:
        print("No cases extracted. Output file left unchanged.")


if __name__ == "__main__":
    main()
