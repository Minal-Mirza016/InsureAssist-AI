# InsureAssist AI 🛡️

**A local, retrieval-augmented insurance document Q&A assistant built with machine learning, FastAPI, Ollama, and Streamlit.**

InsureAssist AI is an educational project that answers questions about a fictional insurance policy. It combines intent classification, semantic document retrieval, and a locally running language model to generate answers with supporting PDF page references.

The project is designed to run locally without a paid LLM API key.

## Features

- **Intent classification:** TF-IDF and Logistic Regression classify insurance-related questions.
- **PDF text extraction:** PyMuPDF extracts text from policy documents.
- **Document chunking:** Text is split into overlapping passages while preserving page numbers and source filenames.
- **Semantic retrieval:** Sentence Transformers generates embeddings using `all-MiniLM-L6-v2`; cosine similarity ranks relevant passages.
- **Retrieval-Augmented Generation (RAG):** Retrieved policy context is passed to a local LLM to generate answers.
- **Local LLM inference:** Llama 3.2 3B runs through Ollama.
- **REST API:** FastAPI exposes intent prediction, document search, and chat endpoints.
- **Interactive interface:** Streamlit provides a chat interface with session-level history and expandable sources.
- **Clickable PDF citations:** Users can open a cited page in the document viewer.
- **No paid LLM API required:** Model inference runs locally through Ollama.

## Architecture

1. The user submits a question through Streamlit.
2. Streamlit sends the question to the FastAPI backend.
3. The backend predicts the question's intent.
4. Semantic retrieval finds relevant passages from the indexed policy documents.
5. The retrieved context and question are sent to Llama 3.2 3B through Ollama.
6. The backend returns the answer, predicted intent, and source references.
7. Streamlit displays the answer and allows the user to open cited PDF pages.

## Technology Stack

- Python
- scikit-learn
- TF-IDF and Logistic Regression
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Cosine similarity
- PyMuPDF
- Ollama and Llama 3.2 3B
- FastAPI
- Streamlit
- Requests
- Joblib
- NumPy and other project dependencies

## Retrieval Evaluation

The initial retrieval evaluation used five sample questions.

| Metric                     | Result |
| -------------------------- | -----: |
| Recall@3                   |   1.00 |
| Mean Reciprocal Rank (MRR) |   1.00 |
| Evaluation questions       |      5 |

These preliminary results indicate that the expected relevant page was retrieved within the top three results for all five questions and ranked first in each case.

**Limitation:** This is a small, controlled evaluation and does not establish generalization to unseen questions or documents. A larger, independently checked evaluation set is needed.

## Project Structure

The repository contains the FastAPI backend, Streamlit frontend, training scripts, retrieval services, sample policy document, and supporting evaluation code.

The exact directory structure may evolve as the project develops.

## Prerequisites

- Python compatible with the installed project dependencies
- Git
- Ollama
- The Ollama model `llama3.2:3b`

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/InsureAssist-AI.git
cd InsureAssist-AI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama if it is not already installed. Download the model once:

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running before testing chat generation.

### 5. Start the FastAPI backend

```powershell
python -m uvicorn main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

### 6. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```powershell
python -m streamlit run frontend/chat_app.py
```

Open the local Streamlit URL printed in the terminal, usually http://localhost:8501.

Keep the backend and Ollama running while using the application.

## Educational Use and Limitations

- The sample policy is fictional and intended for demonstration.
- The application is not an official insurance service and does not provide binding coverage determinations.
- Generated answers may be incorrect or incomplete. Always verify important information against the original policy.
- No real customer documents or confidential insurance data should be uploaded to a public repository.
- The application currently runs locally; it is not a public hosted service.

## Future Improvements

- Expand intent and retrieval evaluation datasets.
- Improve handling of questions not answered by the policy.
- Add automated tests for API, retrieval, and chat functionality.
- Compare semantic retrieval with BM25 and hybrid retrieval where applicable.
- Add response-grounding checks and improve citation verification.
- Improve configuration, logging, and error handling.

## Author

**Minal Mirza**

Computer Science | AI, Machine Learning, and Data Applications
