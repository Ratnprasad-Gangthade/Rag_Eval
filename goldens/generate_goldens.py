import os
import re
import glob
import json
import random

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
load_dotenv(override=True)

# --------------------------------------------------
# Groq model wrapper for DeepEval
# --------------------------------------------------

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

# --------------------------------------------------
# Load VTT files and create chunks
# --------------------------------------------------

def load_chunks():
    texts = []

    for path in glob.glob("data/*.vtt"):
        with open(path, encoding="utf-8") as f:
            lines = [
                ln.strip()
                for ln in f
                if ln.strip()
                and ln.strip() != "WEBVTT"
                and "-->" not in ln
            ]

        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text("\n\n".join(texts))

# --------------------------------------------------
# Load chunks
# --------------------------------------------------

chunks = load_chunks()
print(f"Total chunks available: {len(chunks)}")

# --------------------------------------------------
# Select contexts
# Start with only 5 to avoid Groq TPM limit
# --------------------------------------------------

sample = random.sample(
    chunks,
    min(5, len(chunks))
)

contexts = [[c] for c in sample]
print(f"Generating goldens from {len(contexts)} contexts...")

# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

groq_llm = GroqModel(
    ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )
)

# --------------------------------------------------
# DeepEval Synthesizer
# --------------------------------------------------

synthesizer = Synthesizer(
    model=groq_llm,

    # Keep async enabled because DeepEval expects it,
    # but restrict concurrency to 1.
    async_mode=True,

    # IMPORTANT:
    # Your Groq account has an 8K TPM limit.
    # Running 3 requests simultaneously consumes tokens
    # too quickly.
    max_concurrent=1
)

# --------------------------------------------------
# Generate goldens
# --------------------------------------------------

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)

# --------------------------------------------------
# Convert to your schema
# --------------------------------------------------

rows = []

for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",
    })

# --------------------------------------------------
# Save goldens
# --------------------------------------------------

os.makedirs("goldens", exist_ok=True)

output_file = "goldens/retriever_deepeval_goldens.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )
# --------------------------------------------------
# Final message
# --------------------------------------------------
print()
print(f"wrote {len(rows)} DRAFT goldens -> {output_file}")
print()
print("!! REVIEW EVERY ONE before using:")
print("   - Check grounding")
print("   - Trim padding")
print("   - Fix leading questions")
print()
print("Generation completed successfully.")