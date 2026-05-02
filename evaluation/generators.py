import sys
import time
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    TOGETHER_API_KEY, TOGETHER_BASE_URL,
    TOGETHER_MODEL_PRIMARY, TOGETHER_MODEL_SECONDARY,
)


class RAGGenerator:
    """
    Generates answers using Together AI OpenAI-compatible API.
    Two modes: grounded (uses retrieved passages) and open (no context).
    temperature=0.0 for reproducibility across all calls.
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=TOGETHER_BASE_URL,
        )
        self.delay = 0.15   # seconds between calls (stays under 500 RPM)

    def _call(self, messages: list, model: str,
              max_tokens: int = 300) -> str:
        """Single API call with retry logic."""
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                time.sleep(self.delay)
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                wait = 2.0 * (2 ** attempt)
                print(f"  API error attempt {attempt+1}: {exc}. "
                      f"Waiting {wait:.0f}s...")
                time.sleep(wait)
        return ""

    # Few-shot examples (one English, one Bengali) for grounded prompts
    # Includes a "not found" example to teach the refusal pattern
    _GROUNDED_EXAMPLES = (
        "Passage 1:\nPhotosynthesis is the process by which green plants use sunlight, "
        "water and carbon dioxide to produce food and oxygen.\n\n"
        "Question: What do green plants use to produce food?\n"
        "Answer: Sunlight, water and carbon dioxide.\n\n"
        "Passage 1:\nবাংলাদেশের জাতীয় ফুল হলো শাপলা। এটি জলাশয়ে জন্মায় এবং সাদা রঙের হয়।\n\n"
        "Question: বাংলাদেশের জাতীয় ফুলের নাম কী?\n"
        "Answer: শাপলা।\n\n"
        "Passage 1:\nThe mitochondria is the powerhouse of the cell.\n\n"
        "Question: What is the capital of France?\n"
        "Answer: Not found in context.\n\n"
    )

    # Few-shot examples for open prompts
    _OPEN_EXAMPLES = (
        "Question: What is the capital of Bangladesh?\n"
        "Answer: Dhaka.\n\n"
        "Question: অক্সিজেনের রাসায়নিক প্রতীক কী?\n"
        "Answer: O\n\n"
    )

    def generate_grounded(self, question: str,
                          chunks: list[dict],
                          model: str = None) -> str:
        """Answer using ONLY the provided passages with few-shot examples."""
        model = model or TOGETHER_MODEL_PRIMARY
        passages = ""
        for i, chunk in enumerate(chunks[:5]):
            text = chunk.get("text", "")[:400]
            passages += f"Passage {i+1}:\n{text}\n\n"

        prompt = (
            f"Answer the question using ONLY the information in the passages below.\n"
            f"Rules:\n"
            f"1. Use ONLY words and facts from the passages — do NOT use outside knowledge.\n"
            f"2. Give a short, direct answer. No preamble like 'According to the passage'.\n"
            f"3. Match the language of the question: Bengali question → Bengali answer, "
            f"English question → English answer.\n"
            f"4. If the answer cannot be found in the passages, respond exactly: "
            f"'Not found in context.' — do NOT guess.\n\n"
            f"--- Examples ---\n"
            f"{self._GROUNDED_EXAMPLES}"
            f"--- Now answer this ---\n"
            f"{passages}"
            f"Question: {question}\n"
            f"Answer:"
        )
        return self._call(
            [{"role": "user", "content": prompt}], model
        )

    def generate_open(self, question: str,
                      model: str = None) -> str:
        """Answer without retrieval context, with few-shot examples."""
        model = model or TOGETHER_MODEL_PRIMARY
        prompt = (
            f"Answer the following question with a short, direct answer.\n"
            f"If the question is in Bengali, answer in Bengali. "
            f"If in English, answer in English.\n\n"
            f"--- Examples ---\n"
            f"{self._OPEN_EXAMPLES}"
            f"--- Now answer this ---\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return self._call(
            [{"role": "user", "content": prompt}], model
        )
