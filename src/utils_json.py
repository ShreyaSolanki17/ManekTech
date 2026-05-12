import json
from typing import Any, Optional


def extract_first_json(text: str) -> Optional[str]:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    return text[first : last + 1]


def safe_json_loads(text: str) -> Any:
    snippet = extract_first_json(text) or text
    return json.loads(snippet)
