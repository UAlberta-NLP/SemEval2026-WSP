import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

import random
import requests

# ================================
# 🔑 Configuration
# ================================
API_KEY = "AIVUFuS9Js7QkBJ3RufabHlrKgeNUR4a"
TRAIN_PATH = "data/train.json"
DEV_PATH   = "data/dev.json"
OUTPUT_TSV = "RandomFewShot_output(Mistral-7B)/RandomFewShot(k=200).tsv"

MISTRAL_MODEL = "open-mistral-7b"
TOP_K = 200               # ← number of random few-shot examples
SLEEP_EACH_SEC = 1        # Mistral rate limit: 1 request / sec


# ================================
# 1) Data I/O
# ================================
def load_data(path: str) -> List[Dict[str, Any]]:
    """Load JSON (dict) and return a list of sample objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values())


# ================================
# 2) Few-shot formatting
# ================================
def format_example(ex: Dict[str, Any], i: int) -> str:
    """Format one training example for few-shot guidance."""
    return (
        f"Example {i+1}\n"
        f"Story:\n{ex.get('precontext','')}\n"
        f"Sentence:\n{ex.get('sentence','')}\n"
        f"Homonym:\n{ex.get('homonym','')}\n"
        f"Judged meaning (gold):\n{ex.get('judged_meaning','')}\n"
        f"Gold score (1-5):\n{round(ex.get('average', 3))}\n"
        "----\n"
    )


def build_prompt_with_random_examples(
    train_samples: List[Dict[str, Any]],
    current: Dict[str, Any],
    k: int
) -> str:
    """
    Build few-shot prompt using RANDOM examples, not retrieved ones.
    This serves as a *control baseline*.
    """

    # Select k random training samples (with replacement or without)
    examples = random.sample(train_samples, k=min(k, len(train_samples)))

    header = (
        "You are a grader.\n"
        "Given a short story context, a target homonym, and a candidate sense,\n"
        "rate how plausible the candidate sense is in the story (1=implausible, 5=perfect fit).\n"
        "Use the provided examples as guidance.\n"
        "Output ONLY one integer (1 to 5). No text, no explanation.\n"
    )

    fewshots = "\n".join(format_example(ex, i) for i, ex in enumerate(examples))

    current_block = (
        "Now rate the following item:\n"
        f"Story:\n{current.get('precontext','')}\n"
        f"Sentence:\n{current.get('sentence','')}\n"
        f"Homonym:\n{current.get('homonym','')}\n"
        f"Candidate sense:\n{current.get('judged_meaning','')}\n"
        "Answer with ONE integer (1..5):"
    )

    return f"{header}\n{fewshots}\n{current_block}", [ex.get("sample_id", -1) for ex in examples]


# ================================
# 3) Mistral API
# ================================
class APIMistral:
    """Minimal client for Mistral Chat Completion API."""
    def __init__(self, model_name: str = MISTRAL_MODEL, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def generate(self, prompt: str, temperature=0.1, top_p=0.9, max_tokens=2):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        resp = self.session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ================================
# 4) Main pipeline
# ================================
if __name__ == "__main__":

    train_samples = load_data(TRAIN_PATH)
    dev_samples   = load_data(DEV_PATH)

    os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)

    api = APIMistral(model_name=MISTRAL_MODEL, api_key=API_KEY)

    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("idx\tsample_id\trandom_ids\tllm_text\tllm_score\tfinal_score\n")

        for idx, sample in enumerate(dev_samples):

            # --- RANDOM examples instead of retrieval ---
            prompt, random_ids = build_prompt_with_random_examples(train_samples, sample, TOP_K)

            # Query model
            llm_text = api.generate(prompt)

            try:
                y_llm = int(llm_text.strip())
            except:
                y_llm = 3

            y_llm = max(1, min(5, y_llm))
            final_score = float(y_llm)

            sample_id = sample.get("sample_id", idx)
            clean_llm_text = llm_text.replace("\n", " ").replace("\t", " ")

            f.write(
                f"{idx}\t{sample_id}\t{random_ids}\t"
                f"{clean_llm_text}\t{y_llm:.6f}\t{final_score:.6f}\n"
            )

            print(f"[{idx+1}/{len(dev_samples)}] score={final_score:.3f}  random_k={TOP_K}")

            time.sleep(SLEEP_EACH_SEC)

    print(f"✔ Done. Saved results to {OUTPUT_TSV}")
