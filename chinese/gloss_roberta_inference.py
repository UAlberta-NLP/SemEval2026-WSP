import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_dev_samples(dev_path: Path) -> Dict[str, Dict]:
    with dev_path.open("r") as f:
        return json.load(f)


def highlight_target(context: str, target: str) -> str:
    """
    Wrap occurrences of the target word with double quotes to provide weak supervision.
    """
    pattern = re.compile(rf"\\b{re.escape(target)}\\b", flags=re.IGNORECASE)

    def replacer(match: re.Match) -> str:
        word = match.group(0)
        # Avoid double quoting if already quoted.
        if word.startswith('"') and word.endswith('"'):
            return word
        return f'"{word}"'

    return pattern.sub(replacer, context)


def build_context(sample: Dict) -> str:
    parts = [sample.get("precontext", ""), sample.get("sentence", ""), sample.get("ending", "")]
    # Collapse extra whitespace between the concatenated parts.
    return " ".join(" ".join(parts).split())


def build_gloss(sample: Dict) -> str:
    # Prefix gloss with the target word as weak supervision, e.g., "research: systematic investigation..."
    return f"{sample['homonym']}: {sample['judged_meaning']}"


def pairwise(iterable: List) -> List[List]:
    return [iterable[i : i + 2] for i in range(0, len(iterable), 2)]


def softmax_to_score(prob_first_class: float) -> int:
    # Map p1 in [0,1] to plausibility scores 1..5 via round(1 + 4*p1).
    score = round(1 + 4 * prob_first_class)
    return max(1, min(5, int(score)))


def run_inference(dev_json: str, output_path: str, model_name: str = "roberta-base") -> None:
    data = load_dev_samples(Path(dev_json))
    ids_sorted = sorted(data.keys(), key=lambda x: int(x))
    if len(ids_sorted) % 2 != 0:
        raise ValueError("Expected an even number of samples to form consecutive pairs.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()

    predictions = []
    with torch.no_grad():
        for pair_ids in pairwise(ids_sorted):
            samples = [data[sid] for sid in pair_ids]
            contexts = [highlight_target(build_context(s), s["homonym"]) for s in samples]
            glosses = [build_gloss(s) for s in samples]

            inputs = tokenizer(
                contexts,
                glosses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            for sid, prob in zip(pair_ids, probs):
                p1 = prob[0].item()
                score = softmax_to_score(p1)
                predictions.append({"id": sid, "prediction": score})

    with open(output_path, "w") as f:
        for entry in predictions:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    run_inference("data/dev.json", "gloss_roberta__predictions_dev.jsonl")
