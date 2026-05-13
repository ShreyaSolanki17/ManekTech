from typing import List, Tuple

import chromadb
import google.generativeai as genai
from gemini_client import get_gemini_model

from config import CHROMA_DIR, GEMINI_API_KEY, GEMINI_EMBED_MODEL
from groq_client import call_groq_llm, GROQ_MODEL, GROQ_MODEL_FALLBACK

SYSTEM_PROMPT = """
You are a legal assistant. Answer only from the provided context.
If the answer is not in the context, say you do not know based on the available cases.
Cite case IDs in your answer.
""".strip()


def embed_query(model, query: str) -> List[float]:
    result = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    embedding = result["embedding"] if isinstance(result, dict) else result
    return embedding


def retrieve_context(query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
    model = get_gemini_model()
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    genai.configure(api_key=GEMINI_API_KEY)
    collection = chroma.get_or_create_collection("cases")

    query_emb = embed_query(model, query)
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    return ids, docs


def answer_question(query: str, top_k: int = 3) -> Tuple[str, List[str], List[str]]:
    from groq_client import call_groq_llm, GROQ_MODEL, GROQ_MODEL_FALLBACK
    model = get_gemini_model()
    ids, docs = retrieve_context(query, top_k=top_k)
    context = "\n\n".join(docs)

    prompt = SYSTEM_PROMPT + f"\nQuestion: {query}\n\nContext:\n{context}"
    try:
        response = model.generate_content(prompt)
        return response.text, ids, docs
    except Exception as gemini_exc:
        print(f"Gemini failed: {gemini_exc}\nTrying Groq Llama-3...")
        try:
            groq_response = call_groq_llm(prompt, model=GROQ_MODEL)
            return groq_response, ids, docs
        except Exception as groq_exc:
            print(f"Groq Llama-3 failed: {groq_exc}\nTrying Groq Mixtral...")
            try:
                groq_response2 = call_groq_llm(prompt, model=GROQ_MODEL_FALLBACK)
                return groq_response2, ids, docs
            except Exception as groq2_exc:
                print(f"Groq Mixtral failed: {groq2_exc}")
                raise RuntimeError("All LLMs failed for answer generation.")
