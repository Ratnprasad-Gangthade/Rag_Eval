import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from src.retriever import build_retriever

load_dotenv()

class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.load_model().invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.load_model().ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Groq GPT-OSS-120B"

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "openai/gpt-oss-120b"
THRESHOLD = 0.7

groq_llm = GroqModel(
    ChatGroq(
        model=JUDGE_MODEL,
        temperature=0
    )
)

with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)

goldens = goldens[:5]

retriever = build_retriever()
test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )

for i, test_case in enumerate(test_cases, 1):
    print(f"\nEvaluating test case {i}/{len(test_cases)}...")

    recall_metric = ContextualRecallMetric(
        threshold=THRESHOLD,
        model=groq_llm,
        include_reason=True,
        async_mode=False
    )

    evaluate(
        test_cases=[test_case],
        metrics=[recall_metric],
        hyperparameters={
            "retriever": "base_k5",
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "metric": "contextual_recall",
        },
    )

    print("Contextual Recall completed.")

    precision_metric = ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=groq_llm,
        include_reason=True,
        async_mode=False
    )

    evaluate(
        test_cases=[test_case],
        metrics=[precision_metric],
        hyperparameters={
            "retriever": "base_k5",
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "metric": "contextual_precision",
        },
    )

    print("Contextual Precision completed.")