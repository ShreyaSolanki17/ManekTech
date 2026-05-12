import argparse
import json
from typing import Any, Dict, List

from gemini_client import get_gemini_model
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from config import GEMINI_MODEL
from utils_json import safe_json_loads

SYSTEM_PROMPT = """
You are extracting structured data from Portuguese tax law decisions.
Return ONLY valid JSON with the exact schema fields.
""".strip()

USER_TEMPLATE = """
Extract the following case into JSON with this schema:
{
  "case_id": "string",
  "decision_date": "YYYY-MM-DD",
  "plaintiff_name": "string",
  "defendant_name": "string",
  "court_decision": "Won" | "Lost" | "Settled",
  "summary": "2-3 sentences"
}

Case ID: {case_id}

Raw text:
{raw_text}
""".strip()


class CaseSchema(BaseModel):
    case_id: str
    decision_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    plaintiff_name: str
    defendant_name: str
    court_decision: str
    summary: str


def call_llm(model, case_id: str, raw_text: str) -> Dict[str, Any]:
    prompt = SYSTEM_PROMPT + "\n" + USER_TEMPLATE.format(case_id=case_id, raw_text=raw_text[:12000])
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

    with open(args.output, "w", encoding="utf-8") as f:
        for item in extracted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Extracted {len(extracted)} of {len(records)} cases")


if __name__ == "__main__":
    main()
