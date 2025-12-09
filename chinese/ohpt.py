#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np

DEV_JSON = Path("data/dev.json")
SYNSETS_TSV_EXPANDED = Path("dev-synsets-zh.tsv")
SYNSETS_TSV_RAW = Path("dev-synsets-zh-raw.tsv")
LEMMA_JSONL = Path("dev_chinese_lemma.jsonl")
OUT_PATH_EXPANDED = Path("ohpt_predictions_dev_expanded.jsonl")
OUT_PATH_RAW = Path("ohpt_predictions_dev_raw.jsonl")


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


def get_standard_deviation(l: List[int]) -> float:
    mean = sum(l) / len(l)
    variance = sum((x - mean) ** 2 for x in l) / (len(l) - 1) if len(l) > 1 else 0.0
    return math.sqrt(variance)


def is_within_standard_deviation(prediction: int, labels: List[int]) -> bool:
    avg = sum(labels) / len(labels)
    stdev = get_standard_deviation(labels)
    if (avg - stdev) < prediction < (avg + stdev):
        return True
    if abs(avg - prediction) < 1:
        return True
    return False


def accuracy_within_standard_deviation(preds: List[Tuple[str, int]], gold_data: Dict[str, Dict]) -> float:
    correct, total = 0, 0
    for pid, pred in preds:
        labels = gold_data[str(pid)]["choices"]
        if is_within_standard_deviation(pred, labels):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def calculate_spearman(preds: List[Tuple[str, int]], gold_data: Dict[str, Dict]) -> float:
    gold_list = []
    pred_list = []
    for pid, pred in preds:
        gold_avg = sum(gold_data[str(pid)]["choices"]) / len(gold_data[str(pid)]["choices"])
        gold_list.append(gold_avg)
        pred_list.append(pred)

    corr, _ = spearmanr(pred_list, gold_list)
    return corr


def run_predictions(synsets_tsv: Path, dev_data: Dict, lemmas: List):
    id_to_row = load_synset_rows(synsets_tsv)
    predictions = []

    for doc_id_str in sorted(dev_data.keys(), key=lambda x: int(x)):
        doc_id = int(doc_id_str)
        item = dev_data[doc_id_str]
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

        predictions.append((doc_id_str, score))

    return predictions


def save_predictions(predictions: List[Tuple[str, int]], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for doc_id_str, score in predictions:
            f.write(json.dumps({"id": doc_id_str, "prediction": score}, ensure_ascii=False) + "\n")


def main():
    # Load data
    dev = json.load(DEV_JSON.open(encoding="utf-8"))
    lemmas = load_lemmas(LEMMA_JSONL)

    # Run predictions with both synset files
    print("Running predictions with raw synsets...")
    predictions_raw = run_predictions(SYNSETS_TSV_RAW, dev, lemmas)
    save_predictions(predictions_raw, OUT_PATH_RAW)

    print("Running predictions with manually-expanded synsets...")
    predictions_expanded = run_predictions(SYNSETS_TSV_EXPANDED, dev, lemmas)
    save_predictions(predictions_expanded, OUT_PATH_EXPANDED)

    # Calculate metrics
    acc_raw = accuracy_within_standard_deviation(predictions_raw, dev)
    spearman_raw = calculate_spearman(predictions_raw, dev)

    acc_expanded = accuracy_within_standard_deviation(predictions_expanded, dev)
    spearman_expanded = calculate_spearman(predictions_expanded, dev)

    # Print results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"\nRaw Synsets (dev-synsets-zh-raw.tsv):")
    print(f"  Accuracy (within SD): {acc_raw:.4f}")
    print(f"  Spearman Correlation: {spearman_raw:.4f}")

    print(f"\nManually-Expanded Synsets (dev-synsets-zh.tsv):")
    print(f"  Accuracy (within SD): {acc_expanded:.4f}")
    print(f"  Spearman Correlation: {spearman_expanded:.4f}")

    print(f"\nImprovement:")
    print(f"  Accuracy: +{(acc_expanded - acc_raw):.4f}")
    print(f"  Spearman: +{(spearman_expanded - spearman_raw):.4f}")
    print("="*60)

    # Create grouped bar chart with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Data
    metrics = ['Accuracy', 'Spearman']
    raw_values = [acc_raw, spearman_raw]
    expanded_values = [acc_expanded, spearman_expanded]

    # X positions for bars
    x = np.arange(len(metrics))
    width = 0.35

    # Create bars for Accuracy (on left y-axis)
    bars1_acc = ax1.bar(x[0] - width/2, raw_values[0], width, label='Raw Synsets',
                        color='#FF8C00', alpha=0.8)
    bars2_acc = ax1.bar(x[0] + width/2, expanded_values[0], width, label='Manually-Expanded',
                        color='#4169E1', alpha=0.8)

    # Set up left y-axis for Accuracy
    ax1.set_ylabel('Accuracy', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, max(acc_raw, acc_expanded) * 1.2)

    # Create second y-axis for Spearman
    ax2 = ax1.twinx()

    # Create bars for Spearman (on right y-axis)
    bars1_spear = ax2.bar(x[1] - width/2, raw_values[1], width,
                          color='#FF8C00', alpha=0.8)
    bars2_spear = ax2.bar(x[1] + width/2, expanded_values[1], width,
                          color='#4169E1', alpha=0.8)

    # Set up right y-axis for Spearman
    ax2.set_ylabel('Spearman Correlation', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max(spearman_raw, spearman_expanded) * 1.2)

    # Set x-axis
    ax1.set_xlabel('Metric', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)

    # Add title and legend
    ax1.set_title('OHPT Performance Comparison: Raw vs Manually-Expanded Synsets',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')

    # Tight layout
    fig.tight_layout()

    # Save figure
    plt.savefig('ohpt_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison chart to: ohpt_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
