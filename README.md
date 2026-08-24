# RAG Evaluation Project

A local retrieval-augmented generation (RAG) pipeline for an LLM evaluations course. The project uses VTT lecture transcripts as its knowledge base and evaluates retrieval, generation, and the complete RAG pipeline with DeepEval.

## What it does

The pipeline follows these steps:

1. Loads the VTT files in `data/` and removes timestamps and VTT headers.
2. Splits the transcripts into 1,000-character chunks with 150-character overlap.
3. Embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2` and persists them in Chroma.
4. Retrieves candidates using vector similarity, then reranks them with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
5. Generates a grounded answer with Groq's `openai/gpt-oss-20b` model.
6. Scores the result with DeepEval metrics.

The generator is intentionally faithfulness-first: it is instructed to answer only from the retrieved context and to abstain when the context is insufficient.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access on the first run to download Hugging Face models

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install langchain-huggingface
```

`langchain-huggingface` is installed separately because `src/retriever.py` imports it but it is not currently included in `requirements.txt`.

Create a `.env` file in the repository root:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env`. It is ignored by Git.

## Quick start

Run the retriever smoke test:

```powershell
python -m src.retriever
```

Run the complete RAG smoke test:

```powershell
python -m src.rag_pipeline
```

The first retriever run creates the local `chroma_store/` directory. Later runs reuse it. To rebuild the store after changing the transcripts or chunking settings, delete `chroma_store/` and run a smoke test again.

## Evaluation commands

Run the full RAG evaluation:

```powershell
python -m evals.eval_rag_pipeline
```

This evaluates the golden queries using:

- Contextual relevancy
- Faithfulness
- Answer relevancy

Run retriever-only evaluation:

```powershell
python -m evals.eval_retriever
```

This evaluates contextual recall and contextual precision against the first five records in `goldens/retriever_goldens.json`.

Run generator-only evaluation:

```powershell
python -m evals.eval_generator
```

This evaluates the first five records in `goldens/faithfulness_dataset.json` using faithfulness and answer relevancy. The generator test uses each record's `ideal_context`, so it isolates answer generation from retrieval quality.

All evaluation scripts make live Groq requests. They use a threshold of `0.7`; the full pipeline also processes cases one at a time and waits between batches to reduce rate-limit errors. Evaluation can therefore take several minutes and incur API usage.

## Generate retriever goldens

To create a draft DeepEval retriever dataset from randomly selected transcript chunks:

```powershell
python goldens/generate_goldens.py
```

The script writes `goldens/retriever_deepeval_goldens.json`. Review and verify every generated record before using it for evaluation; generated records are marked with `source: "TODO-verify"`.

## Project layout

```text
data/                         VTT lecture transcripts
chroma_store/                 Persistent Chroma vector store
src/retriever.py              Loading, chunking, embedding, and retrieval
src/reranker.py               Cross-encoder reranking
src/generator.py              Groq generation and grounding prompt
src/rag_pipeline.py           Retrieval plus generation orchestration
evals/eval_retriever.py       Retriever metrics
evals/eval_generator.py       Generator metrics
evals/eval_rag_pipeline.py   End-to-end RAG metrics
goldens/                      Curated and generated evaluation datasets
resources/                    DeepEval learning/example script
requirements.txt              Python dependencies
```

## Notes

- Run commands from the repository root because paths such as `data/`, `goldens/`, and `chroma_store/` are relative paths.
- The embedding and reranker models are downloaded and cached locally by Hugging Face on first use.
- `chroma_store/` and DeepEval cache files are generated artifacts and are ignored by Git.
- `pytest` is included as a dependency, but this repository currently contains script-based evaluations rather than a conventional `tests/` suite.