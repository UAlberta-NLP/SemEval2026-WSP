#!/usr/bin/env python3
import csv
import json
from pathlib import Path

DEV_JSON = Path("data/dev.json")
SYNSETS_TSV = Path("dev-synsets-zh.tsv")
LEMMA_JSONL = Path("dev_chinese_lemma.jsonl")
OUT_PATH = Path("ohpt_predictions_dev.jsonl")


def normalize_gloss(text: str) -> str:
    return text.strip().strip('"').lower()


def tokenize_targets(text: str):
    # Target synset columns may contain multiple lemmas and asterisks; strip and split.
    cleaned = text.replace("*", " ")
    return {tok for tok in cleaned.split() if tok}


def load_synset_rows(path: Path):
    id_to_row = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids = [int(part) for part in row["instance numbers"].split()]
            row["_instance_ids"] = ids
            for inst_id in ids:
                id_to_row[inst_id] = row
    return id_to_row


def load_lemmas(path: Path):
    lemmas = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            lemmas.append(json.loads(line))
    return lemmas


def main():
    dev = json.load(DEV_JSON.open(encoding="utf-8"))
    id_to_row = load_synset_rows(SYNSETS_TSV)
    lemmas = load_lemmas(LEMMA_JSONL)

    predictions = []
    for doc_id_str in sorted(dev.keys(), key=lambda x: int(x)):
        doc_id = int(doc_id_str)
        item = dev[doc_id_str]
        row = id_to_row.get(doc_id)
        predicted_lemma = lemmas[doc_id]["predicted_chinese_lemma"] if doc_id < len(lemmas) else ""
        score = 1

        if row:
            judged = normalize_gloss(item["judged_meaning"])
            gloss1 = normalize_gloss(row["gloss 1"])
            gloss2 = normalize_gloss(row["gloss 2"])
            target_text = None

            if judged == gloss1:
                target_text = row["Target synset 1"]
            elif judged == gloss2:
                target_text = row["Target synset 2"]

            if target_text and predicted_lemma:
                target_tokens = tokenize_targets(target_text)
                if predicted_lemma in target_tokens:
                    score = 5

        predictions.append({"id": doc_id_str, "prediction": score})

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
