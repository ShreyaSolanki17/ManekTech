import argparse
import json
from typing import List

import chromadb
from gemini_client import get_gemini_model
from tqdm import tqdm

from config import CHROMA_DIR


def load_cases(path: str) -> List[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def build_documents(cases: List[dict]) -> List[str]:
    docs = []
    for c in cases:
        doc = (
            f"Case {c['case_id']}\n"
            f"Decision date: {c['decision_date']}\n"
            f"Plaintiff: {c['plaintiff_name']}\n"
            f"Defendant: {c['defendant_name']}\n"
            f"Decision: {c['court_decision']}\n"
            f"Summary: {c['summary']}"
        )
        docs.append(doc)
    return docs


def embed_texts(model, texts: List[str]) -> List[List[float]]:
    # Gemini's embedding API: batch call
    embeddings = model.embed_content(texts)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cases.jsonl")
    parser.add_argument("--persist", default=CHROMA_DIR)
    parser.add_argument("--collection", default="cases")
    args = parser.parse_args()

    cases = load_cases(args.input)
    docs = build_documents(cases)

    model = get_gemini_model()
    chroma = chromadb.PersistentClient(path=args.persist)
    collection = chroma.get_or_create_collection(args.collection)

    batch_size = 64
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        batch_cases = cases[i : i + batch_size]
        batch_docs = docs[i : i + batch_size]
        embeddings = embed_texts(model, batch_docs)

        collection.add(
            ids=[c["case_id"] for c in batch_cases],
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=[
                {
                    "decision_date": c["decision_date"],
                    "court_decision": c["court_decision"],
                }
                for c in batch_cases
            ],
        )


if __name__ == "__main__":
    main()
