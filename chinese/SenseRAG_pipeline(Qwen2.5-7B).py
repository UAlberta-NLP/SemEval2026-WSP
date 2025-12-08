import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer

# ================================
# 🔑 Configuration
# ================================
TRAIN_PATH = "data/train.json"
DEV_PATH   = "data/dev.json"
OUTPUT_TSV = "SenseRAG_output(Qwen2.5-7B)/SenseRAG_output(k=6).tsv"

EMBED_MODEL = "all-MiniLM-L6-v2"
QWEN_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 6
SLEEP_EACH_SEC = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# 1) Data I/O
# ================================
def load_data(path: str) -> List[Dict[str, Any]]:
    """Load JSON dict and return list of sample objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values())


# ================================
# 2) Embedding utilities
# ================================
def build_index(train_samples: List[Dict[str, Any]], model_name: str):
    """Build embedding index for retrieval with Sentence-BERT."""
    encoder = SentenceTransformer(model_name)
    texts = [f"{s.get('precontext','')} {s.get('sentence','')}".strip()
             for s in train_samples]
    embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return encoder, embeddings


def retrieve_topk(query_sample, train_samples, encoder, train_embeds, top_k=TOP_K):
    """Retrieve top-K neighbors based on cosine similarity."""
    q_text = f"{query_sample.get('precontext','')} {query_sample.get('sentence','')}".strip()
    q_vec = encoder.encode([q_text], convert_to_numpy=True)[0]

    sims = (train_embeds @ q_vec) / (
        np.linalg.norm(train_embeds, axis=1) * (np.linalg.norm(q_vec) + 1e-12)
    )
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [train_samples[i] for i in top_idx]


# ================================
# 3) Few-shot formatting
# ================================
def format_example(ex, i):
    """Format one retrieved sample into a few-shot block."""
    return (
        f"Example {i+1}\n"
        f"Story:\n{ex.get('precontext','')}\n"
        f"Sentence:\n{ex.get('sentence','')}\n"
        f"Homonym:\n{ex.get('homonym','')}\n"
        f"Judged meaning (gold):\n{ex.get('judged_meaning','')}\n"
        f"Gold score (1-5):\n{round(ex.get('average', 3))}\n"
        "----\n"
    )


def build_prompt_with_fewshots(examples, current):
    """Construct the final grading prompt."""
    rubic = (
        "Scoring rules:\n"
        "5 = perfect fit; strongly supported by story.\n"
        "4 = good fit; clearly plausible.\n"
        "3 = possible but ambiguous.\n"
        "2 = unlikely; weak support.\n"
        "1 = impossible or contradicts context.\n\n"
    )

    header = (
        "You are a precise semantic grader.\n"
        "Rate how well the candidate sense fits the story.\n"
        + rubic +
        "Output ONLY a single digit from 1 to 5.\n"
    )

    fewshots = "\n".join(format_example(ex, i) for i, ex in enumerate(examples))

    current_block = (
        "Now rate the following item:\n"
        f"Story:\n{current.get('precontext','')}\n"
        f"Sentence:\n{current.get('sentence','')}\n"
        f"Homonym:\n{current.get('homonym','')}\n"
        f"Candidate sense:\n{current.get('judged_meaning','')}\n"
        "Answer with ONE integer (1..5)."
    )

    return f"{header}\n{fewshots}\n{current_block}"


# ================================
# 4) Qwen Local Inference Client (FULLY FIXED)
# ================================
class QwenLocal:

    def __init__(self, model_name=QWEN_MODEL):
        """Load tokenizer and model."""
        print("Loading Qwen2.5 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

    def generate(self, prompt, max_tokens=8):
        """Generate a score and correctly extract last assistant segment."""
        messages = [{"role": "user", "content": prompt}]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)

        output_ids = self.model.generate(
            model_inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
        )

        # Do NOT skip special tokens — needed for parsing assistant block
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # Extract the last assistant message between <|im_start|>assistant ... <|im_end|>
        import re
        matches = re.findall(
            r"<\|im_start\|>assistant\s*(.*?)(?=<\|im_end\|>)",
            decoded,
            flags=re.S
        )

        if matches:
            last_answer = matches[-1].strip()
        else:
            last_answer = decoded.strip()

        # Extract only a digit 1–5
        m = re.search(r"[1-5]", last_answer)
        if m:
            return m.group(0)

        return "3"  # fallback


# ================================
# 5) Main Pipeline
# ================================
if __name__ == "__main__":

    train_samples = load_data(TRAIN_PATH)
    dev_samples   = load_data(DEV_PATH)

    encoder, train_embeds = build_index(train_samples, EMBED_MODEL)
    api = QwenLocal(QWEN_MODEL)

    os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)

    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("idx\tsample_id\tretrieved_ids\tllm_text\tllm_score\tfinal_score\n")

        for idx, sample in enumerate(dev_samples):

            neighbors = retrieve_topk(sample, train_samples, encoder, train_embeds, top_k=TOP_K)
            retrieved_ids = [ex.get("sample_id", -1) for ex in neighbors]

            prompt = build_prompt_with_fewshots(neighbors, sample)
            llm_text = api.generate(prompt)
            print(llm_text)

            try:
                y_llm = int(llm_text.strip())
            except:
                y_llm = 3

            y_llm = max(1, min(5, y_llm))
            final_score = float(y_llm)

            sample_id = sample.get("sample_id", idx)

            f.write(
                f"{idx}\t{sample_id}\t{retrieved_ids}\t"
                f"{llm_text}\t{y_llm:.6f}\t{final_score:.6f}\n"
            )

            print(f"[{idx+1}/{len(dev_samples)}] score={final_score:.3f} retrieved={retrieved_ids}")

            time.sleep(SLEEP_EACH_SEC)

    print(f"✅ Done. Saved results to {OUTPUT_TSV}")
