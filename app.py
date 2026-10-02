"""Streamlit chat frontend for the course-material RAG pipeline."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


@st.cache_resource(show_spinner="Loading retrieval index and reranker...")
def get_pipeline():
    from src.rag_pipeline import RagPipeline

    return RagPipeline()


def answer_question(pipeline, question):
    """Adapt the backend response for the chat UI without exposing raw context."""
    result = pipeline.invoke(question)
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }


def main():
    st.set_page_config(page_title="Course Notes", layout="centered")
    st.markdown(
        """
        <style>
        :root {
            --ink: #172b2a;
            --muted: #657775;
            --accent: #177d70;
            --line: #dce6e2;
            --canvas: #f4f8f6;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(145deg, #f4f8f6 0%, #f8f7f1 58%, #f1f7f8 100%);
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] { max-width: 850px; padding-top: 2.5rem; }
        h1 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.78);
        }
        [data-testid="stChatInput"] { border-color: var(--line); }
        .app-kicker {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Course Notes")
        st.caption("LLM Evaluation")
        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="app-kicker">RAG assistant</div>', unsafe_allow_html=True)
    st.title("Ask about LLM evaluation")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                sessions = ", ".join(f"Session {source}" for source in message["sources"])
                st.caption(f"Retrieved from: {sessions}")

    question = st.chat_input("Ask a question about the course")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not os.getenv("OPENAI_API_KEY"):
        error_message = "Add OPENAI_API_KEY to the project .env file, then restart the app."
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching course material and composing an answer..."):
            try:
                response = answer_question(get_pipeline(), question)
            except Exception:
                error_message = (
                    "The request could not be completed. Check the API key, network "
                    "connection, and local retrieval index, then try again."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
                return

        st.markdown(response["answer"])
        if response["sources"]:
            sessions = ", ".join(f"Session {source}" for source in response["sources"])
            st.caption(f"Retrieved from: {sessions}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        }
    )


if __name__ == "__main__":
    main()