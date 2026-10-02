import os
import tempfile
import unittest
from contextlib import chdir
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "local-test-key")

from app import answer_question
from src import rag_pipeline
from src.retriever import load_transcripts


class FakeDocument:
    def __init__(self, page_content, session):
        self.page_content = page_content
        self.metadata = {"session": session}


class FakeRetriever:
    def invoke(self, query):
        return [
            FakeDocument(f"Context for {query}", "3"),
            FakeDocument("Additional context", "5"),
            FakeDocument("Another passage from session 3", "3"),
        ]


class AppIntegrationTests(unittest.TestCase):
    def test_chat_adapter_uses_existing_pipeline_and_returns_citations(self):
        retriever = FakeRetriever()
        with (
            patch.object(rag_pipeline, "RerankingRetriever", return_value=retriever),
            patch.object(rag_pipeline, "generate", return_value="Grounded answer."),
        ):
            pipeline = rag_pipeline.RagPipeline()
            response = answer_question(pipeline, "What is faithfulness?")

        self.assertEqual(response, {
            "answer": "Grounded answer.",
            "sources": ["3", "5"],
        })

    def test_transcript_loading_does_not_depend_on_launch_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with chdir(temporary_directory):
                documents = load_transcripts()

        self.assertEqual(len(documents), 8)
        self.assertEqual(
            {document.metadata["session"] for document in documents},
            {str(session) for session in range(1, 9)},
        )
        self.assertTrue(all(document.page_content for document in documents))


if __name__ == "__main__":
    unittest.main()