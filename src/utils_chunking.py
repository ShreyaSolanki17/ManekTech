from typing import List

def chunk_text(text: str, max_tokens: int = 4000, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of approximately max_tokens (by words), with optional overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks
