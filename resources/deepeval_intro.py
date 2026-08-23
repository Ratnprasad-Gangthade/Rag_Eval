from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq

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

groq_llm = GroqModel(
    ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )
)

case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=groq_llm,
    include_reason=True
)

evaluate(
    test_cases=[case_1, case_2],
    metrics=[metric]
)