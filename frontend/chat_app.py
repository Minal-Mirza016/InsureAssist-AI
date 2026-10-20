
from pathlib import Path

import requests
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/chat"

DOCUMENT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "documents"
)


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="InsureAssist AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ InsureAssist AI")
st.caption(
    "Your local insurance policy assistant powered by machine learning, "
    "document retrieval, and a local language model."
)

st.info(
    "This educational assistant uses a fictional sample policy. "
    "It does not make official coverage or claim decisions."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1


def select_pdf_page(source: str, page: int) -> None:
    """Select a PDF and the page to display."""
    st.session_state.selected_source = source
    st.session_state.selected_page = page


# --------------------------------------------------
# Main layout: PDF LEFT, CHAT RIGHT
# --------------------------------------------------

pdf_col, chat_col = st.columns([1.1, 1], gap="large")


# ==================================================
# LEFT COLUMN: PDF VIEWER
# ==================================================

with pdf_col:
    st.subheader("📖 Policy document viewer")

    selected_source = st.session_state.selected_source
    selected_page = st.session_state.selected_page

    if selected_source:
        st.caption(
            f"Document: {selected_source} · Page {selected_page}"
        )

        if st.button("← Close PDF"):
            st.session_state.selected_source = None
            st.session_state.selected_page = 1
            st.rerun()

        # Ensure the filename cannot reference another directory.
        safe_filename = Path(selected_source).name

        if (
            safe_filename != selected_source
            or not safe_filename.lower().endswith(".pdf")
        ):
            st.error("Invalid PDF filename.")

        else:
            pdf_path = DOCUMENT_DIR / safe_filename

            if not pdf_path.is_file():
                st.error(f"PDF not found: {safe_filename}")

            else:
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()

                    pdf_viewer(
                        input=pdf_bytes,
                        width=700,
                        height=800,
                        pages_to_render=[selected_page],
                    )

                except Exception as exc:
                    st.error(f"Unable to display the PDF: {exc}")

    else:
        st.info(
            "The PDF will appear here when you click a page citation "
            "under a chatbot answer."
        )


# ==================================================
# RIGHT COLUMN: CHAT AND SOURCES
# ==================================================

with chat_col:
    st.subheader("💬 Chat")

    # Display the conversation history.
    for message_index, message in enumerate(
        st.session_state.messages
    ):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("intent"):
                st.caption(
                    f"Detected intent: {message['intent']}"
                )

            sources = message.get("sources", [])

            if sources:
                with st.expander(
                    "📚 View supporting policy sources"
                ):
                    for source_index, source in enumerate(sources):
                        filename = source.get("source", "")
                        page = int(source.get("page", 1))
                        similarity = source.get("similarity")

                        st.markdown(f"**{filename}**")

                        citation_label = f"📄 Open PDF — Page {page}"

                        if similarity is not None:
                            citation_label += (
                                f" · Similarity: {similarity}"
                            )

                        if st.button(
                            citation_label,
                            key=(
                                f"citation_{message_index}_"
                                f"{source_index}"
                            ),
                            use_container_width=True,
                        ):
                            select_pdf_page(filename, page)
                            st.rerun()

    # User input.
    question = st.chat_input(
        "Ask about claims, coverage, premiums, or renewals..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:
            with st.spinner(
                "Searching policy documents and generating an answer..."
            ):
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "top_k": 3,
                    },
                    timeout=240,
                )

                response.raise_for_status()
                data = response.json()

                answer = data.get(
                    "answer",
                    "I couldn't generate an answer.",
                )

                intent = data.get(
                    "predicted_intent",
                    "unknown",
                )

                sources = data.get("sources", [])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "intent": intent,
                        "sources": sources,
                    }
                )

        except requests.exceptions.Timeout:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The request took too long. Please try again. "
                        "The local language model may need more time."
                    ),
                }
            )

        except requests.exceptions.ConnectionError:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I couldn't connect to the FastAPI backend. "
                        "Make sure the backend is running at "
                        "http://127.0.0.1:8000."
                    ),
                }
            )

        except requests.exceptions.HTTPError as exc:
            detail = ""

            try:
                detail = response.json().get("detail", "")
            except (ValueError, AttributeError):
                pass

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"The backend returned an error: {exc}. {detail}"
                    ),
                }
            )

        except requests.exceptions.RequestException as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Request failed: {exc}",
                }
            )

        except (ValueError, KeyError, TypeError) as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"The backend response could not be processed: {exc}"
                    ),
                }
            )

        # Refresh the page to display the new messages.
        # Notice: we do NOT automatically open a PDF here.
        st.rerun()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("🛡️ About InsureAssist AI")

    st.write(
        "InsureAssist AI combines an ML intent classifier, "
        "semantic document retrieval, and a locally running LLM."
    )

    st.divider()

    st.caption(
        "Educational demonstration only. Consult the official "
        "policy and insurer for actual coverage decisions."
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_source = None
        st.session_state.selected_page = 1
        st.rerun()
