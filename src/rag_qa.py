from typing import List, Tuple

import chromadb
from gemini_client import get_gemini_model

from config import CHROMA_DIR, GEMINI_MODEL

SYSTEM_PROMPT = """
You are a legal assistant. Answer only from the provided context.
If the answer is not in the context, say you do not know based on the available cases.
Cite case IDs in your answer.
""".strip()


def embed_query(model, query: str) -> List[float]:
    # Gemini's embedding API
    return model.embed_content([query])[0]


def retrieve_context(query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
    model = get_gemini_model()
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection("cases")

    query_emb = embed_query(model, query)
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    return ids, docs


def answer_question(query: str, top_k: int = 3) -> Tuple[str, List[str]]:
    model = get_gemini_model()
    ids, docs = retrieve_context(query, top_k=top_k)
    context = "\n\n".join(docs)

    prompt = SYSTEM_PROMPT + f"\nQuestion: {query}\n\nContext:\n{context}"
    response = model.generate_content(prompt)

    return response.text, ids
