# RAG Evaluation Notes

These notes explain the concepts implemented in this project. The central idea is simple:

> A RAG application must retrieve useful context and generate an answer that is correct, relevant, and grounded in that context.

## 1. End-to-end flow

```text
VTT transcripts
      |
      v
Clean text -> chunks -> embeddings -> Chroma vector store
                                      |
User question -> similarity retrieval -> cross-encoder reranking
                                      |
                                      v
                         Top context chunks
                                      |
                                      v
                              Groq generator
                                      |
                                      v
                                  Answer
                                      |
                                      v
                    DeepEval evaluation metrics
```

The application can fail in two main places:

1. The retriever returns irrelevant or incomplete context.
2. The generator receives good context but ignores it or invents information.

Both parts must work well for the complete RAG system to work well.

## 2. Why evaluate an LLM application?

### Why

LLM output is probabilistic. The same question can produce different wording, different reasoning, or a different answer on different runs. A few manual questions, sometimes called vibe testing, cannot reliably show whether an application is ready for production.

### What

Evaluation is a repeatable process that measures application quality against defined criteria and a known dataset.

### How

Create test cases, run the application on them, calculate metrics, inspect failures, and compare the results with a baseline.

### Example

Instead of asking a chatbot five questions and deciding that it "feels good," save 50 representative questions and check every new version for retrieval quality, faithfulness, and answer relevance.

## 3. RAG: Retrieval-Augmented Generation

### Why

An LLM's internal knowledge may be incomplete, outdated, or unrelated to a private document collection. Retrieval gives the model relevant external context at query time.

### What

RAG is a pattern in which a retriever finds relevant documents and a generator uses those documents to answer a question.

### How

1. Convert source documents into searchable chunks.
2. Retrieve chunks related to the user's question.
3. Place the chunks in the model prompt as context.
4. Generate an answer based on that context.

### Example

For the question "What is regression testing in LLM evaluation?", the retriever finds transcript chunks discussing regression testing. The generator uses those chunks to explain the concept instead of relying only on its general training.

## 4. Loading and cleaning VTT transcripts

### Why

VTT files contain timestamps and formatting markers that are useful for video playback but add noise to semantic search.

### What

The loader reads every `.vtt` file in `data/`, removes `WEBVTT`, timestamp lines, and blank lines, then joins the remaining text.

### How

`src/retriever.py` uses `load_transcripts()` to create LangChain `Document` objects. Each document keeps the session number in metadata.

### Example

```text
WEBVTT
00:01:10.000 --> 00:01:15.000
What is vibe testing?
```

becomes searchable text similar to:

```text
What is vibe testing?
```

## 5. Chunking

### Why

A complete lecture may be too large to retrieve or place in an LLM prompt. Smaller chunks make matching more precise and keep the generation context manageable.

### What

Chunking divides a document into smaller pieces. This project uses 1,000-character chunks with 150 characters of overlap.

### How

`RecursiveCharacterTextSplitter` creates the chunks. The overlap repeats a small part of the previous chunk so that a concept split at a boundary is less likely to lose its surrounding context.

### Example

If a definition begins near the end of chunk A and continues at the start of chunk B, the 150-character overlap gives chunk B some of the definition's beginning.

## 6. Embeddings

### Why

Keyword matching can miss questions that use different words from the source. Embeddings allow the system to compare meaning rather than only exact words.

### What

An embedding is a numeric representation of text. Texts with similar meaning are placed near one another in vector space.

### How

This project uses `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`. It embeds transcript chunks when the store is built and embeds each query during retrieval.

### Example

The query "Why do models sometimes invent facts?" may retrieve a chunk discussing hallucination even when the word "invent" does not appear in the chunk.

## 7. Chroma vector store

### Why

Recomputing embeddings for every query is slow. A vector store persists embeddings and makes similarity searches reusable.

### What

Chroma is the local vector database used by this project. Its data is stored in `chroma_store/`.

### How

On the first run, `load_store()` embeds the transcript chunks and persists them. On later runs, it opens the existing store instead of rebuilding it.

### Example

After changing the transcripts or chunking settings, delete `chroma_store/` and run `python -m src.retriever` so the store is rebuilt with the new data.

## 8. Similarity retrieval

### Why

The generator should receive only the context most likely to help answer the question. Irrelevant context can confuse the model and reduce faithfulness.

### What

Similarity retrieval returns the chunks whose embeddings are closest to the query embedding. The base retriever returns the top five chunks, or `k=5`.

### How

`build_retriever()` creates a Chroma retriever with `search_kwargs={"k": 5}`.

### Example

For "What is recall at k?", the retriever should return chunks defining recall and top-k retrieval, not unrelated chunks about benchmark saturation.

## 9. Reranking

### Why

Vector similarity is fast but can be approximate. A second model can examine the query and each candidate together and make a more precise relevance decision.

### What

Reranking sorts an over-retrieved candidate list and keeps the strongest results.

### How

`RerankingRetriever` retrieves 10 candidates, scores each query-chunk pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`, sorts by score, and keeps the top five.

### Example

Similarity retrieval may return 10 chunks about evaluation. The cross-encoder can place the two chunks specifically answering "What is contextual recall?" above the other eight.

## 10. Grounded generation

### Why

A fluent answer is not necessarily a correct answer. The model must use the retrieved evidence and avoid adding unsupported facts.

### What

Grounded generation produces an answer from the question and retrieved context. The generator in `src/generator.py` is instructed to answer only from the course context and abstain when the context is insufficient.

### How

`RagPipeline.invoke()` retrieves documents, extracts their text, and passes the question plus context to `generate()`. The Groq model runs with temperature `0` to reduce unnecessary variation.

### Example

If the context explains faithfulness but says nothing about pricing, a pricing question should receive:

```text
I don't have enough information in the course material to answer that.
```

## 11. The RAG triad

### Why

One overall score cannot explain why a RAG application failed. The question, retrieved context, and answer each have a different relationship that should be evaluated separately.

### What

The RAG triad contains three checks:

| Check | Main question |
| --- | --- |
| Context relevancy | Is the retrieved context relevant to the question? |
| Faithfulness | Is the answer supported by the retrieved context? |
| Answer relevancy | Does the answer address the question? |

### How

The full evaluator creates an `LLMTestCase` with the input, generated answer, and retrieval context, then measures all three DeepEval metrics.

### Example

An answer can be faithful but still fail answer relevancy if it accurately discusses chunking when the user asked about reranking. A relevant answer can fail faithfulness if it adds facts that were not in the context.

## 12. Golden dataset

### Why

Evaluation needs stable test cases. Without a fixed set of questions and expected information, results cannot be compared across application versions.

### What

A golden dataset contains representative questions and trusted answers or contexts. This repository has:

- `goldens/retriever_goldens.json`: questions, ideal answers, and source sessions for retriever evaluation.
- `goldens/faithfulness_dataset.json`: questions, ideal contexts, and source sessions for generator and pipeline evaluation.

### How

Run the application against each golden question, then compare the output or retrieved context with the expected information using suitable metrics.

### Example

```json
{
  "id": "g012",
  "query": "What is regression testing in LLM evaluation?",
  "ideal_answer": "...",
  "source": "Session 9"
}
```

## 13. Reference-based and reference-free evaluation

### Why

Different tasks have different definitions of correctness. Some have one known answer; others allow multiple valid answers.

### What

- **Reference-based:** a golden answer or expected output exists and the result is compared with it.
- **Reference-free:** no fixed answer exists; a human or judge model scores the result using a rubric.

### How

Use exact or programmatic comparison where outputs are deterministic. Use rubric-based human or LLM judgment where correct answers can have different wording or structure.

### Example

Comparing two SQL queries character by character is weak because different queries can return the same table. Comparing their result sets is a better reference-based check. For an explanatory answer, an LLM judge can assess whether it is supported and relevant without requiring identical wording.

## 14. Retriever evaluation

### Why

The generator cannot answer well if the required evidence was never retrieved. Retriever evaluation isolates this first stage of the system.

### What

This project measures:

- **Contextual recall:** how much of the relevant information was retrieved.
- **Contextual precision:** how much of the retrieved information was relevant.

### How

`evals/eval_retriever.py` retrieves context for the first five retriever goldens and sends each case to DeepEval with a threshold of `0.7`.

### Example

If a question needs two relevant chunks and the retriever returns only one, recall is approximately 50%. If it returns five chunks but only two are relevant, precision is lower than if four of the five are relevant.

## 15. Generator evaluation

### Why

Retrieval and generation are separate failure points. Testing generation with known-good context shows whether the model can use evidence correctly.

### What

The generator evaluator measures:

- **Faithfulness:** whether claims in the answer are supported by the supplied context.
- **Answer relevancy:** whether the answer addresses the question.

### How

`evals/eval_generator.py` uses each golden record's `ideal_context` directly, calls `generate()`, and evaluates the generated answer.

### Example

If the ideal context says that online evaluation runs on live production traffic, an answer claiming that it runs only before deployment should receive a low faithfulness score.

## 16. LLM as a judge

### Why

String matching is too strict for natural-language answers. Two answers can use different words and still have the same meaning, while a matching phrase can appear in an incorrect answer.

### What

An LLM judge reads the question, answer, context, and evaluation criteria, then returns a score and reason.

### How

DeepEval calls the Groq judge model for metrics such as faithfulness, contextual relevancy, and answer relevancy. The evaluators use a threshold of `0.7`.

### Example

These answers are semantically equivalent:

```text
Online evaluation checks behavior on live production traffic.
Online eval measures a deployed system as real users interact with it.
```

An LLM judge can recognize that equivalence even though the strings are different.

## 17. Component, pipeline, and application evaluation

### Why

Testing at several levels makes failures easier to locate and gives a more complete view of quality.

### What

- **Component evaluation:** tests one part, such as retrieval or generation.
- **Pipeline evaluation:** tests the complete RAG path from question to answer.
- **Application evaluation:** tests the user-facing system, including operational behavior and safety where applicable.

### How

Use `eval_retriever.py` for retrieval, `eval_generator.py` for generation, and `eval_rag_pipeline.py` for the combined RAG pipeline.

### Example

If retriever recall is high but pipeline faithfulness is low, the likely problem is generation or prompting rather than document search.

## 18. Regression testing

### Why

An update can improve one metric while silently damaging another. A change should be compared with the previous baseline before deployment.

### What

Regression testing means rerunning the evaluation suite against a new application version and checking whether quality improved or declined.

### How

Save the scores from the current version as a baseline. After changing the prompt, embedding model, chunk size, reranker, or LLM, run the same golden dataset and compare every metric.

### Example

Changing chunk overlap might increase contextual recall but reduce precision. Regression testing exposes both changes instead of judging the update from one successful demo.

## 19. Golden generation workflow

### Why

Creating high-quality test cases manually takes time. Synthetic generation can provide a starting point, especially when a document collection is large.

### What

`goldens/generate_goldens.py` samples transcript chunks, asks a Groq model to generate questions and expected answers, and writes draft records to `goldens/retriever_deepeval_goldens.json`.

### How

1. Load and chunk the VTT transcripts.
2. Randomly select up to five chunks.
3. Generate one golden per selected context.
4. Review every generated record.
5. Correct grounding, wording, and source metadata before evaluation.

### Example

An automatically generated question may be grammatically valid but impossible to answer from its selected chunk. Such a record must be edited or removed before it becomes a trusted golden.

## 20. Rate limits and reproducibility

### Why

Evaluation uses multiple model calls. Sending too many requests or tokens at once can trigger Groq rate limits and make results incomplete.

### What

Rate limiting is a service constraint that temporarily rejects requests after a usage threshold is exceeded.

### How

The evaluators reduce concurrency, insert delays, and retry some failed requests. The full evaluator uses a batch size of one and waits between batches.

### Example

If a run receives a 429 response, wait for the retry logic to finish or rerun later. Do not interpret a partially completed run as a complete evaluation result.

## 21. Useful commands

Run from the repository root:

```powershell
# Inspect retrieved chunks
python -m src.retriever

# Run one complete RAG smoke test
python -m src.rag_pipeline

# Evaluate retrieval
python -m evals.eval_retriever

# Evaluate generation
python -m evals.eval_generator

# Evaluate the complete pipeline
python -m evals.eval_rag_pipeline

# Generate draft retriever goldens
python goldens/generate_goldens.py
```

## 22. Practical debugging checklist

When a result is poor, inspect the stages in this order:

1. **Source data:** Is the answer actually present in the VTT transcripts?
2. **Chunking:** Was the relevant explanation split awkwardly?
3. **Retrieval:** Did the top five chunks contain the needed evidence?
4. **Reranking:** Did the best candidate survive the reranking step?
5. **Prompting:** Did the generator receive the context in the expected format?
6. **Generation:** Did the answer stay grounded and address every part of the question?
7. **Evaluation:** Was the metric, golden answer, and judge model appropriate?

This order moves from evidence availability to retrieval, then generation and measurement, making failures easier to diagnose.