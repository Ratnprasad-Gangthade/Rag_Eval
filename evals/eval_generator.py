import os
import json
import time

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from groq import RateLimitError

from src.generator import generate

load_dotenv(override=True)

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
GROQ_MODEL = "openai/gpt-oss-20b"
THRESHOLD = 0.7
MAX_TEST_CASES = 5
REQUEST_DELAY = 5
MAX_RETRIES = 5


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                return self.model.invoke(prompt).content
            except RateLimitError:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("Groq rate limit persisted after multiple retries.")

    async def a_generate(self, prompt: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                return (await self.model.ainvoke(prompt)).content
            except RateLimitError:
                import asyncio
                await asyncio.sleep(10 * (attempt + 1))
        raise RuntimeError("Groq rate limit persisted after multiple retries.")

    def get_model_name(self):
        return f"Groq {GROQ_MODEL}"


groq_llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    max_tokens=2048,
    api_key=os.getenv("GROQ_API_KEY")
)

judge_model = GroqJudge(groq_llm)

with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)[:MAX_TEST_CASES]

test_cases = []

for g in goldens:
    context = g["ideal_context"]
    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=generate(g["query"], context),
            retrieval_context=context
        )
    )
    time.sleep(REQUEST_DELAY)

faithfulness_metric = FaithfulnessMetric(
    threshold=THRESHOLD,
    model=judge_model,
    include_reason=True,
    async_mode=False
)

relevancy_metric = AnswerRelevancyMetric(
    threshold=THRESHOLD,
    model=judge_model,
    include_reason=True,
    async_mode=False
)

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 60}\nTEST CASE {i}/{len(test_cases)}\n{'=' * 60}")
    print(f"Question: {test_case.input}")
    print(f"\nGenerated Answer: {test_case.actual_output}")

    try:
        faithfulness_metric.measure(test_case)
        faithfulness = faithfulness_metric.score
        print(f"\nFaithfulness: {faithfulness:.4f}")
        print(f"Reason: {faithfulness_metric.reason}")

        time.sleep(REQUEST_DELAY)

        relevancy_metric.measure(test_case)
        relevancy = relevancy_metric.score
        print(f"\nAnswer Relevancy: {relevancy:.4f}")
        print(f"Reason: {relevancy_metric.reason}")

        results.append({
            "test_case": i,
            "query": test_case.input,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy
        })

    except (RateLimitError, ValueError) as e:
        print(f"Evaluation failed: {e}")

    time.sleep(REQUEST_DELAY)

if results:
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    for r in results:
        print(
            f"Test {r['test_case']}: "
            f"Faithfulness={r['faithfulness']:.4f}, "
            f"Relevancy={r['answer_relevancy']:.4f}"
        )

    print(f"\nAverage Faithfulness: {avg_faithfulness:.4f}")
    print(f"Average Answer Relevancy: {avg_relevancy:.4f}")
    print(f"{'=' * 60}")