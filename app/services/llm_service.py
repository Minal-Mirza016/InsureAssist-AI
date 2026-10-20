
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using retrieved policy passages.
    Ollama runs the language model locally.
    """

    prompt = f"""
You are InsureAssist AI, an educational insurance policy assistant.

Answer the user's question using ONLY the policy context below.

Rules:
1. Do not invent policy rules, benefits, amounts, or deadlines.
2. If the context does not contain enough information, say so clearly.
3. Distinguish information stated in the sample policy from assumptions.
4. Cite the source filename and page number provided in the context.
5. Do not claim that an insurance claim is approved or guaranteed.
6. Keep the answer clear and helpful.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Write the answer using the policy context. Include source citations.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            return (
                "I couldn't generate an answer. "
                "Please try asking again."
            )

        return answer

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running "
            "and that the local API is available at localhost:11434."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Ollama took too long to respond. Please try again."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama returned an HTTP error: {exc}"
        ) from exc
