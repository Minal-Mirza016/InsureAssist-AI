
from pathlib import Path

import fitz
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = PROJECT_DIR / "data" / "documents"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "document_index.joblib"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DOCUMENT_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENT_DIR}"
        )

    records = []

    # Extract each PDF page separately so page citations are preserved.
    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_text = page.get_text("text")

                if not page_text.strip():
                    print(
                        f"Warning: no extractable text on "
                        f"page {page_index + 1} of {pdf_path.name}"
                    )
                    continue

                chunks = split_text(page_text)

                for chunk_number, chunk in enumerate(chunks, start=1):
                    records.append(
                        {
                            "source": pdf_path.name,
                            "page": page_index + 1,
                            "chunk_id": (
                                f"{pdf_path.stem}_"
                                f"p{page_index + 1}_c{chunk_number}"
                            ),
                            "text": chunk,
                        }
                    )

    if not records:
        raise ValueError(
            "No text chunks were extracted from the PDFs."
        )

    print(f"\nExtracted {len(records)} chunks.")

    # Download/load the embedding model once, then encode all chunks.
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [record["text"] for record in records]

    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Save metadata and vectors together.
    index = {
        "model_name": EMBEDDING_MODEL_NAME,
        "records": records,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
    }

    joblib.dump(index, OUTPUT_PATH)

    print("\nIndex created successfully.")
    print(f"PDF files indexed: {len(pdf_files)}")
    print(f"Chunks indexed: {len(records)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved index: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
