
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_DIR / "data" / "processed" / "document_index.joblib"

_encoder = None
_index = None


def load_index():
    """Load the saved document index and its matching embedding model."""
    global _encoder, _index

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Document index not found: {INDEX_PATH}. "
            "Run training/build_document_index.py first."
        )

    if _index is None:
        _index = joblib.load(INDEX_PATH)

    if _encoder is None:
        _encoder = SentenceTransformer(_index["model_name"])

    return _encoder, _index


def search_documents(question: str, top_k: int = 3):
    """Return the most semantically similar document passages."""
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    encoder, index = load_index()

    records = index["records"]
    embeddings = index["embeddings"]

    query_embedding = encoder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Because both query and document vectors are normalized,
    # their dot product equals cosine similarity.
    scores = np.dot(embeddings, query_embedding)

    # Rank the passages by similarity, highest first.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for position in ranked_indices:
        record = records[int(position)]

        results.append(
            {
                "source": record["source"],
                "page": record["page"],
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "similarity": round(float(scores[position]), 4),
            }
        )

    return results
