import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from gemini_client import get_gemini_model
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


def call_llm(model, case_id: str, raw_text: str) -> Dict[str, Any]:
    # Escape braces in raw_text to keep prompt formatting safe
    safe_raw_text = raw_text[:12000].replace('{', '{{').replace('}', '}}')
    prompt = PROMPT_TEMPLATE.replace("<<CASE_ID>>", case_id).replace("<<RAW_TEXT>>", safe_raw_text)
    response = model.generate_content(prompt)
    return safe_json_loads(response.text)


def extract_cases(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model = get_gemini_model()
    results: List[Dict[str, Any]] = []

    for rec in tqdm(records, desc="Extracting JSON"):
        for attempt in range(2):
            try:
                data = call_llm(model, rec["case_id"], rec["raw_text"])
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
