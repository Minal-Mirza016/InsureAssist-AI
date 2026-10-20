from pathlib import Path
from fastapi.responses import FileResponse
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.services.retrieval_service import search_documents
from app.services.llm_service import generate_answer

# ---------------------------------------------
# 1. Locate and load the trained ML model
# ---------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "intent_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run training/train_intent_model.py first."
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------
# 2. Create the FastAPI application
# ---------------------------------------------

app = FastAPI(
    title="InsureAssist AI API",
    description=(
        "An educational insurance chatbot API "
        "with a trained ML intent classifier."
    ),
    version="1.0.0",
)


# ---------------------------------------------
# 3. Define the expected request format
# ---------------------------------------------

class IntentRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=1000,
        description="The user's insurance-related question",
        examples=["How do I submit an insurance claim?"],
    )
class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)

# ---------------------------------------------
# 4. Health-check endpoint
# ---------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to InsureAssist AI!",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ---------------------------------------------
# 5. Predict the intent of a question
# ---------------------------------------------

@app.post("/predict-intent")
def predict_intent(request: IntentRequest):
    question = request.text.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace only.",
        )

    # Predict the intent category
    predicted_intent = model.predict([question])[0]

    # Calculate the model's confidence estimate
    probabilities = model.predict_proba([question])[0]
    confidence = float(max(probabilities))

    return {
        "question": question,
        "predicted_intent": str(predicted_intent),
        "confidence": round(confidence, 4),
        "note": (
            "Confidence is a model estimate, not a guarantee "
            "that the prediction is correct."
        ),
    }
@app.post("/search")
def search_policy_documents(request: SearchRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        results = search_documents(
            question=question,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return {
        "question": question,
        "results_count": len(results),
        "results": results,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace only.",
        )

    # 1. Predict the question's intent.
    predicted_intent = str(model.predict([question])[0])

    # 2. Retrieve relevant passages from the sample policy.
    try:
        results = search_documents(
            question=question,
            top_k=request.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    if not results:
        return {
            "question": question,
            "predicted_intent": predicted_intent,
            "answer": (
                "I couldn't find relevant information in the "
                "available policy documents."
            ),
            "sources": [],
        }

    # 3. Format evidence, preserving document and page citations.
    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}, "
            f"Page: {result['page']}\n"
            f"Passage: {result['text']}"
        )

    context = "\n\n".join(context_parts)

    # 4. Generate an answer using the local LLM.
    try:
        answer = generate_answer(question, context)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    # 5. Return the answer and its supporting sources.
    sources = [
        {
            "source": result["source"],
            "page": result["page"],
            "similarity": result["similarity"],
            "chunk_id": result["chunk_id"],
        }
        for result in results
    ]

    return {
        "question": question,
        "predicted_intent": predicted_intent,
        "answer": answer,
        "sources": sources,
    }

DOCUMENT_DIR = Path(__file__).resolve().parent / "data" / "documents"


@app.get("/documents/{filename}")
def get_document(filename: str):
    # Only allow a simple filename, not a path to another folder.
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    pdf_path = DOCUMENT_DIR / filename

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="PDF document not found.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline",
    )