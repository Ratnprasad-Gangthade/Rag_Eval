import os
import json
import time

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from src.rag_pipeline import RagPipeline

load_dotenv(override=True)

# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = "goldens/faithfulness_dataset.json"

GROQ_MODEL = "openai/gpt-oss-20b"

THRESHOLD = 0.7

# Groq TPM limit from your error
TPM_LIMIT = 8000

# Number of test cases evaluated at once
BATCH_SIZE = 1

# Wait between batches
BATCH_DELAY = 15

# Retry configuration for 429 errors
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10


# ============================================================
# GROQ JUDGE
# ============================================================

class GroqJudge(DeepEvalBaseLLM):

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the environment."
            )

        self.model = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=1024,
            api_key=api_key,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt):

        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES):

            try:
                response = self.model.invoke(prompt)

                return response.content

            except Exception as e:

                error_message = str(e)

                # Handle Groq 429 rate limit
                if "429" in error_message or "RateLimitError" in error_message:

                    print(
                        f"\n⚠️ Groq rate limit reached."
                        f"\nRetry {attempt + 1}/{MAX_RETRIES}"
                        f"\nWaiting {retry_delay} seconds..."
                    )

                    time.sleep(retry_delay)

                    retry_delay *= 2

                else:
                    raise

        raise RuntimeError(
            "Groq rate limit persisted after maximum retries."
        )

    async def a_generate(self, prompt):

        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES):

            try:

                response = await self.model.ainvoke(prompt)

                return response.content

            except Exception as e:

                error_message = str(e)

                # Handle Groq 429 rate limit
                if "429" in error_message or "RateLimitError" in error_message:

                    print(
                        f"\n⚠️ Groq rate limit reached."
                        f"\nRetry {attempt + 1}/{MAX_RETRIES}"
                        f"\nWaiting {retry_delay} seconds..."
                    )

                    await self._async_sleep(retry_delay)

                    retry_delay *= 2

                else:
                    raise

        raise RuntimeError(
            "Groq rate limit persisted after maximum retries."
        )

    async def _async_sleep(self, seconds):

        import asyncio

        await asyncio.sleep(seconds)

    def get_model_name(self):

        return f"Groq {GROQ_MODEL}"


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:

    goldens = json.load(f)


print(f"\n📚 Loaded {len(goldens)} golden test cases.")


# ============================================================
# INITIALIZE RAG
# ============================================================

print("\n🔄 Initializing RAG pipeline...")

rag = RagPipeline()

print("✅ RAG pipeline initialized.")


# ============================================================
# INITIALIZE JUDGE
# ============================================================

print("\n🤖 Initializing Groq judge...")

judge = GroqJudge()

print(f"✅ Judge model: {GROQ_MODEL}")


# ============================================================
# CREATE TEST CASES
# ============================================================

test_cases = []

print("\n🔍 Running RAG pipeline...\n")

for i, g in enumerate(goldens):

    print(
        f"Processing RAG test case "
        f"{i + 1}/{len(goldens)}..."
    )

    result = rag.invoke(g["query"])

    test_case = LLMTestCase(
        input=g["query"],
        actual_output=result["answer"],
        retrieval_context=result["context"],
    )

    test_cases.append(test_case)

    # Small delay to avoid stressing the RAG/embedding side
    time.sleep(2)


print(
    f"\n✅ Created {len(test_cases)} DeepEval test cases."
)


# ============================================================
# DEFINE METRICS
# ============================================================

metrics = [

    ContextualRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    ),

    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    ),

    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    ),
]


# ============================================================
# RUN EVALUATION IN SMALL BATCHES
# ============================================================

total_cases = len(test_cases)

print("\n" + "=" * 60)
print("STARTING DEEPEVAL EVALUATION")
print("=" * 60)

print(f"Total test cases : {total_cases}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Metrics          : {len(metrics)}")
print(f"Groq model       : {GROQ_MODEL}")
print("=" * 60)


for start in range(0, total_cases, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        total_cases
    )

    batch = test_cases[start:end]

    print(
        f"\n\n🚀 Evaluating test cases "
        f"{start + 1}-{end} / {total_cases}"
    )

    try:

        evaluate(
            test_cases=batch,
            metrics=metrics,
        )

        print(
            f"\n✅ Batch {start + 1}-{end} completed."
        )

    except Exception as e:

        error_message = str(e)

        # If Groq still gives a 429 at the DeepEval level,
        # wait and retry the entire batch.
        if "429" in error_message or "RateLimitError" in error_message:

            print(
                "\n⚠️ Groq rate limit detected at DeepEval level."
            )

            print(
                "Waiting 30 seconds before retry..."
            )

            time.sleep(30)

            print(
                f"🔁 Retrying batch {start + 1}-{end}..."
            )

            evaluate(
                test_cases=batch,
                metrics=metrics,
            )

        else:

            print(
                f"\n❌ Evaluation failed for "
                f"batch {start + 1}-{end}"
            )

            raise

    # Wait between batches
    if end < total_cases:

        print(
            f"\n⏳ Waiting {BATCH_DELAY} seconds "
            "before next batch..."
        )

        time.sleep(BATCH_DELAY)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("🎉 EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)