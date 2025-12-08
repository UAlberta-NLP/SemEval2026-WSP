import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

import requests
from sentence_transformers import SentenceTransformer

# ================================
# 🔑 Configuration
# ================================
API_KEY = "AIVUFuS9Js7QkBJ3RufabHlrKgeNUR4a"   # your existing key
TRAIN_PATH = "data/train.json"
DEV_PATH   = "data/dev.json"
OUTPUT_TSV = "SenseRAG_output/SenseRAG_output(k=200).tsv"
EMBED_MODEL = "all-MiniLM-L6-v2"
MISTRAL_MODEL = "open-mistral-7b"
TOP_K = 200                  # ← adjustable: 1 (single-shot) or higher (multi-shot)
SLEEP_EACH_SEC = 1       # obey 1 req/s

# ================================
# 1) Data I/O
# ================================
def load_data(path: str) -> List[Dict[str, Any]]:
    """Load a JSON dict file where each value is a sample object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values())

# ================================
# 2) Embedding utilities
# ================================
def build_index(train_samples: List[Dict[str, Any]],
                model_name: str = EMBED_MODEL):
    """
    Build an index for retrieval using SentenceTransformer.
    We embed (precontext + sentence) for each training sample.
    """
    encoder = SentenceTransformer(model_name)
    texts = [f"{s.get('precontext','')} {s.get('sentence','')}".strip()
             for s in train_samples]
    embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return encoder, embeddings

def retrieve_topk(query_sample: Dict[str, Any],
                  train_samples: List[Dict[str, Any]],
                  encoder: SentenceTransformer,
                  train_embeds: np.ndarray,
                  top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve top-K most similar training samples to the query sample,
    using cosine similarity on (precontext + sentence) embeddings.
    """
    q_text = f"{query_sample.get('precontext','')} {query_sample.get('sentence','')}".strip()
    q_vec = encoder.encode([q_text], convert_to_numpy=True)[0]
    sims = (train_embeds @ q_vec) / (np.linalg.norm(train_embeds, axis=1) * (np.linalg.norm(q_vec) + 1e-12))
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [train_samples[i] for i in top_idx]

# ================================
# 3) Few-shot formatting (support k>1)
# ================================
def format_example(ex: Dict[str, Any], i: int) -> str:
    """Format one retrieved example as a compact few-shot block."""
    return (
        f"Example {i+1}\n"
        f"Story:\n{ex.get('precontext','')}\n"
        f"Sentence:\n{ex.get('sentence','')}\n"
        f"Homonym:\n{ex.get('homonym','')}\n"
        f"Judged meaning (gold):\n{ex.get('judged_meaning','')}\n"
        f"Gold score (1-5):\n{round(ex.get('average', 3))}\n"
        "----\n"
    )

def build_prompt_with_fewshots(examples: List[Dict[str, Any]], current: Dict[str, Any]) -> str:
    """
    Build the final prompt:
    - include up to TOP_K retrieved examples as few-shot demonstrations;
    - append the current item for grading;
    - require a single integer output (1..5).
    """
    header = (
        "You are a grader.\n"
        "Given a short story context, a target homonym, and a candidate sense,\n"
        "rate how plausible the candidate sense is in the story (1=implausible, 5=perfect fit).\n"
        "Use the provided examples as guidance for the rating scale.\n"
        "Output ONLY a single integer from 1 to 5. No text, no JSON, no explanation.\n"
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

    return f"{header}\n{fewshots}\n{current_block}"

# ================================
# 4) Mistral API client
# ================================
class APIMistral:
    """Minimal client for Mistral Chat Completions API."""
    def __init__(self, model_name: str = MISTRAL_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.mistral.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def generate(self, prompt: str, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 2) -> str:
        """Send prompt to Mistral and return the raw output."""
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
# 5) Main pipeline
# ================================
if __name__ == "__main__":
    train_samples = load_data(TRAIN_PATH)
    dev_samples   = load_data(DEV_PATH)

    encoder, train_embeds = build_index(train_samples, EMBED_MODEL)
    api = APIMistral(model_name=MISTRAL_MODEL, api_key=API_KEY)

    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("idx\tsample_id\tretrieved_ids\tllm_text\tllm_score\tfinal_score\n")

        for idx, sample in enumerate(dev_samples):
            # Retrieve top-K neighbors
            neighbors = retrieve_topk(sample, train_samples, encoder, train_embeds, top_k=TOP_K)
            retrieved_ids = [ex.get("sample_id", -1) for ex in neighbors]

            # Build prompt including all retrieved examples
            prompt = build_prompt_with_fewshots(neighbors, sample)

            # Query LLM
            llm_text = api.generate(prompt, temperature=0.1, top_p=0.9, max_tokens=2)
            try:
                y_llm = int(llm_text.strip())
            except Exception:
                y_llm = 3
            y_llm = max(1, min(5, y_llm))
            final_score = float(y_llm)

            sample_id = sample.get("sample_id", idx)
            clean_llm_text = (llm_text or "").replace("\t", " ").replace("\n", " ")
            f.write(
                f"{idx}\t{sample_id}\t{retrieved_ids}\t"
                f"{clean_llm_text}\t{float(y_llm):.6f}\t{final_score:.6f}\n"
            )

            print(f"[{idx+1}/{len(dev_samples)}] score={final_score:.3f} (retrieved={retrieved_ids})")
            time.sleep(SLEEP_EACH_SEC)

    print(f"✅ Done. Saved results to {OUTPUT_TSV}")
