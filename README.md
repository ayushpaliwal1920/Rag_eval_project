# RAG Evaluation Project

A retrieval-augmented teaching assistant for course material on LLM evaluations, with an offline evaluation and regression-testing suite. The Streamlit frontend uses the existing Python RAG pipeline directly; the repository does not currently expose an HTTP server or REST API.

## Features

- Answers questions using eight course-session VTT transcripts as its knowledge source.
- Retrieves transcript chunks from a persistent Chroma vector store using OpenAI embeddings.
- Reranks retrieved chunks with a SentenceTransformers CrossEncoder before generation.
- Generates answers with a context-grounded OpenAI chat model and supports streaming at the generator layer.
- Shows the retrieved session numbers in the chat UI without exposing raw transcript chunks.
- Evaluates retrieval, generation, the RAG pipeline, application quality, safety, latency, cost, and reliability.
- Saves evaluation snapshots and compares candidates against a baseline with PASS, REVIEW, or FAIL outcomes.
- Optionally scores LangSmith production traces with the online-evaluation worker.

## Architecture and Workflow

```text
Student question
	|
	v
Streamlit app (app.py)
	|
	v
RagPipeline.invoke(query)
	|
	+--> RerankingRetriever
	|       +--> OpenAI text-embedding-3-large
	|       +--> persistent Chroma similarity search (fetch_k)
	|       +--> CrossEncoder reranking (top_k)
	|
	+--> grounded LangChain prompt -> OpenAI gpt-4o-mini
	|
	v
Answer and retrieved session identifiers

Offline evals -> snapshot JSON -> baseline comparison -> verdict
Optional LangSmith traces -> evals/eval_online.py -> feedback scores
```

Transcript files are loaded from `data/`, split into overlapping chunks, embedded on first use, and stored in `chroma_store/`. Subsequent app starts reuse that local vector store. The pipeline response contains `query`, `context`, `answer`, and `sources`; the chat frontend displays only the answer and source sessions.

The offline suite builds one pipeline and reuses it across its evaluations. This keeps measurements tied to one retriever/reranker configuration. Quality and safety scores are judged with DeepEval metrics; operational evaluations measure latency, estimated token cost, and retry-based reliability. `evals/metric_registry.py` defines how changes affect the final regression verdict.

## Tech Stack

- Python 3.12+
- Streamlit
- LangChain, LangChain OpenAI, and LangChain Chroma
- ChromaDB with OpenAI `text-embedding-3-large` embeddings
- SentenceTransformers `cross-encoder/ms-marco-MiniLM-L-6-v2`
- OpenAI `gpt-4o-mini` for answer generation and evaluation judges
- DeepEval for LLM application evaluation
- LangSmith for optional tracing and online feedback
- `uv` for dependency and lockfile management

## Project Structure

```text
app.py                         Streamlit chat frontend
src/
  generator.py                 Grounded answer generation and streaming
  rag_pipeline.py              Retrieval-plus-generation pipeline
  reranker.py                  Chroma retrieval and CrossEncoder reranking
  retriever.py                 VTT ingestion, chunking, embeddings, vector store
evals/
  run_suite.py                 Full evaluation and snapshot writer
  compare.py                   Baseline/candidate regression verdict
  eval_retriever.py            Retriever precision and recall
  eval_generator.py            Generator faithfulness and answer relevance
  eval_rag_pipeline.py         RAG triad metrics
  eval_application.py          Correctness, completeness, and style
  eval_safety.py               Scope, information leakage, and toxicity
  eval_ops.py                  Latency, estimated cost, and reliability
  eval_online.py               Optional LangSmith trace scorer
  harness.py                   Shared golden loading and score summaries
  metric_registry.py           Regression metric rules and tolerances
goldens/                       Evaluation datasets and reference answers
data/                          Source VTT course transcripts
chroma_store/                  Generated persistent vector index (local)
baselines/                     Generated baseline and candidate snapshots (local)
pyproject.toml                 Project metadata and dependencies
uv.lock                        Locked dependency resolution
.env.example                   Environment variable template
```

The older standalone evaluation scripts remain available for their individual concerns; `evals/run_suite.py` is the consolidated regression entry point.

## Setup and Installation

Prerequisites: Python 3.12, `uv`, an OpenAI API key, and network access for model APIs and the first-time CrossEncoder model download.

1. Create a local environment file and add a valid OpenAI key:

   ```bash
  cp -n .env.example .env
  # Add OPENAI_API_KEY to .env (preserve any existing settings)
   ```

2. Resolve and install the locked dependencies:

   ```bash
   uv sync
   ```

The local `.env`, Chroma index, and generated evaluation snapshots are excluded from version control. Never commit real API keys.

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | OpenAI chat completions, embeddings, and DeepEval model judging. |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | No | LangSmith API key when tracing or the online scorer is enabled. |
| `LANGCHAIN_PROJECT` | No | LangSmith project name. The online scorer currently queries `cx doubt solver`. |

The current pipeline uses OpenAI integrations. A `GOOGLE_API_KEY` by itself does not configure these components.

## Run the Backend

The backend is a Python module rather than a separate HTTP service. From the repository root, run a single request smoke example:

```bash
uv run python -m src.rag_pipeline
```

On first use, the retriever embeds the transcripts and creates `chroma_store/`; this may take a while. Later runs reuse the local index. The command calls OpenAI and may incur API charges.

## Run the Streamlit Frontend

From the repository root after setting `OPENAI_API_KEY` and installing dependencies:

```bash
uv run streamlit run app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`. Ask course-related questions in the chat input. Use **New conversation** in the sidebar to clear the current conversation.

## Python API Usage

The backend API is the `RagPipeline` Python class. There is no HTTP endpoint or `curl` API in the current repository.

```python
from src.rag_pipeline import RagPipeline

pipeline = RagPipeline()
result = pipeline.invoke("What is faithfulness in a RAG pipeline?")

print(result["answer"])
print(result["sources"])  # e.g. ["3", "5"]
```

`result["context"]` contains the retrieved transcript chunks for internal application use and evaluation. Avoid displaying or logging raw context unnecessarily.

## Evaluation and Regression

Run the full offline suite and write a candidate snapshot:

```bash
uv run python -m evals.run_suite --label "candidate description"
```

Create or replace a baseline intentionally:

```bash
uv run python -m evals.run_suite --baseline --label "baseline description"
```

Compare the default baseline and candidate snapshots:

```bash
uv run python -m evals.compare
```

The suite writes `baselines/baseline.json` or `baselines/candidate.json`; use `--out path.json` for a custom location. `--full` includes informational metrics in the snapshot, `--quiet` reduces operational/safety output, and `evals.compare --all` shows informational metrics. The full suite calls the configured LLMs and can incur API cost.

The separate standalone modules can also be run with `uv run python -m evals.<module>`, for example `uv run python -m evals.eval_retriever`. The online scorer is a long-running worker and requires LangSmith tracing and a populated `cx doubt solver` project:

```bash
uv run python -m evals.eval_online
```

## Example Usage

```text
Question: What does faithfulness measure in a RAG pipeline?

Answer: Faithfulness checks whether the generated answer is supported by the
retrieved context, rather than introducing unsupported claims.

Retrieved from: Session 3
```

The actual response depends on the retrieved transcript chunks and the model output.

## Screenshots

<!-- Add a screenshot of the Streamlit chat here. -->

## Future Improvements

- Add a dedicated HTTP API if independent backend deployment or external clients are needed.
- Stream generated response chunks in the Streamlit chat for lower perceived latency.
- Add automated tests for evaluation snapshot classification and metric aggregation.
- Add controls for model, retrieval depth, and cost/latency policy through centralized configuration.
- Add authenticated deployment, observability dashboards, and production-grade rate limiting.
- Review and promote only verified golden examples; some repository datasets are drafts or carry `TODO-verify` provenance.
