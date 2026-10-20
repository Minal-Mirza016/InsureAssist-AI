# InsureAssist AI 🛡️

**A local, retrieval-augmented insurance document Q&A assistant built with Machine Learning, LLMs, FastAPI, and Streamlit.**

InsureAssist AI is an educational project that answers questions about a fictional insurance policy. It combines intent classification, semantic document retrieval, and a locally running Large Language Model (LLM) to generate answers with supporting PDF page references.

The application runs locally using **Llama 3.2 3B through Ollama**, without requiring a paid LLM API key.

## ✨ Features

- **Intent Classification:** TF-IDF and Logistic Regression classify insurance-related questions.
- **PDF Text Extraction:** PyMuPDF extracts text from policy documents.
- **Document Chunking:** Extracted text is divided into overlapping passages while preserving source filenames and page numbers.
- **Semantic Retrieval:** Sentence Transformers generates embeddings using `all-MiniLM-L6-v2`, and cosine similarity ranks relevant passages.
- **Retrieval-Augmented Generation (RAG):** Retrieved policy passages provide context for LLM-generated answers.
- **Local LLM Inference:** Llama 3.2 3B runs locally through Ollama.
- **REST API:** FastAPI exposes endpoints for intent prediction, document search, and chat.
- **Interactive Chat Interface:** Streamlit provides a conversational interface with session-level chat history.
- **Clickable PDF Citations:** Users can open cited policy pages in the PDF viewer alongside the chat.
- **Local Execution:** No paid LLM API key is required.

## 🏗️ Architecture

1. The user submits a question through the Streamlit interface.
2. Streamlit sends the question to the FastAPI backend.
3. The backend predicts the question's intent using the saved TF-IDF and Logistic Regression pipeline.
4. Semantic retrieval identifies relevant passages from the indexed policy document.
5. The question and retrieved context are passed to Llama 3.2 3B through Ollama.
6. FastAPI returns the generated answer, predicted intent, and supporting source references.
7. Streamlit displays the response and allows the user to open cited PDF pages.

## 🧰 Technology Stack

| Component                   | Technology                  |
| --------------------------- | --------------------------- |
| Programming language        | Python                      |
| Intent classification       | TF-IDF, Logistic Regression |
| ML framework                | scikit-learn                |
| Embedding model             | `all-MiniLM-L6-v2`          |
| Retrieval ranking           | Cosine similarity           |
| PDF processing              | PyMuPDF                     |
| Local LLM runtime           | Ollama                      |
| Generative model            | Llama 3.2 3B                |
| Backend API                 | FastAPI                     |
| Frontend                    | Streamlit                   |
| HTTP requests               | Requests                    |
| Model and index persistence | Joblib                      |

## 📊 Retrieval Evaluation

The initial retrieval evaluation used five sample questions.

| Metric                     | Result |
| -------------------------- | -----: |
| Recall@3                   |   1.00 |
| Mean Reciprocal Rank (MRR) |   1.00 |
| Evaluation questions       |      5 |

In this initial test set, the expected relevant page appeared within the top three retrieved results for all five questions and ranked first each time.

**Evaluation limitation:** These preliminary results come from a small, controlled sample. They do not establish generalization to unseen questions or documents. A larger test set with independently verified relevance labels is needed.

## 📁 Project Structure

```text
InsureAssist-AI/
├── app/
│   └── services/
│       ├── llm_service.py
│       └── retrieval_service.py
├── data/
│   ├── documents/
│   │   └── sample_insurance_policy.pdf
│   └── processed/
│       └── document_index.joblib
├── frontend/
│   └── chat_app.py
├── models/
│   └── intent_pipeline.joblib
├── results/
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── training/
│   ├── build_document_index.py
│   ├── check_dataset.py
│   ├── create_sample_policy.py
│   ├── evaluate_retrieval.py
│   ├── intent_dataset.csv
│   └── train_intent_model.py
├── main.py
├── requirements.txt
├── test_llm.py
├── test_model.py
├── test_retrieval.py
├── .gitignore
└── README.md
```

_This structure reflects the files currently identified in the project. Update it if the repository contains additional files or if the structure changes._

## ⚙️ Prerequisites

- Python installed and compatible with the project's dependencies
- Git
- Ollama
- The Ollama model `llama3.2:3b`

The initial model download requires internet access. After downloading the dependencies and model, the application can perform inference locally.

## 🚀 Run Locally

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

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

If PowerShell blocks script activation, use Git Bash instead:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Install Ollama if necessary, then download the model:

```powershell
ollama pull llama3.2:3b
```

Ensure Ollama is running before asking the chatbot to generate answers.

### 5. Start the FastAPI backend

In the first terminal:

```powershell
python -m uvicorn main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### 6. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```powershell
python -m streamlit run frontend/chat_app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

Keep the FastAPI backend and Ollama running while using the application.

### 7. Prepare the document index if needed

The repository includes a processed document index for the sample policy. If you need to rebuild it after changing the policy or if the index is missing, run the project's indexing script:

```powershell
python -m training.build_document_index
```

Verify that the script completes successfully and generates the index expected by the retrieval service.

## 🧪 Tests

The repository includes scripts for testing the LLM integration, intent classifier, and document retrieval:

```powershell
python test_model.py
python test_retrieval.py
python test_llm.py
```

Some tests may require the model, dependencies, or local services to be available. These scripts are project test utilities; they are not necessarily a complete automated test suite.

## 🔒 Educational Use and Limitations

- The sample policy is fictional and intended for demonstration.
- The application is not an official insurance service and does not provide binding coverage determinations.
- Generated answers may be incorrect or incomplete. Verify important information against the original policy.
- Retrieval similarity does not guarantee that a passage supports the generated answer.
- No real customer documents, credentials, or confidential insurance information should be committed to a public repository.
- The application currently runs locally and is not a publicly hosted service.

## 🔮 Future Improvements

- Expand intent classification and retrieval evaluation datasets.
- Improve handling of questions that cannot be answered from the policy.
- Add automated tests for API responses, retrieval, and error handling.
- Compare semantic retrieval with BM25 and hybrid retrieval.
- Improve answer grounding and citation verification.
- Add configuration management, structured logging, and more robust error handling.

## 👩‍💻 Author

**Minal Mirza**

Computer Science | Artificial Intelligence | Machine Learning | Data Applications
