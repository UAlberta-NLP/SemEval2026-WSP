import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# ---------- Preprocessing (Sent-CLS variant) ----------
def build_context(sample: Dict) -> str:
    parts = [sample.get("precontext", ""), sample.get("sentence", ""), sample.get("ending", "")]
    return " ".join(" ".join(parts).split())


def build_gloss(sample: Dict) -> str:
    # Sent-CLS variant does not prepend the target word to the gloss.
    return sample["judged_meaning"]


def softmax_to_score(prob_first_class: float) -> int:
    score = round(1 + 4 * prob_first_class)
    return max(1, min(5, int(score)))


# ---------- Evaluation logic (mirrors evaluate.py) ----------
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


# ---------- Dataset ----------
class GlossDataset(Dataset):
    def __init__(self, data: Dict[str, Dict]):
        self.ids = sorted(data.keys(), key=lambda x: int(x))
        self.data = data

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        sample = self.data[sid]
        context = build_context(sample)
        gloss = build_gloss(sample)
        # Soft label mapped to [0,1]
        target_prob = (float(sample["average"]) - 1.0) / 4.0
        return sid, context, gloss, target_prob


def collate_fn(batch, tokenizer, device):
    ids, contexts, glosses, targets = zip(*batch)
    enc = tokenizer(
        list(contexts),
        list(glosses),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return (
        list(ids),
        {k: v.to(device) for k, v in enc.items()},
        torch.tensor(targets, dtype=torch.float, device=device),
    )


# ---------- Training / Evaluation ----------
def evaluate(model, dataloader, gold_data, device) -> Tuple[float, float, List[Tuple[str, int]]]:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for ids, inputs, _targets in dataloader:
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            p1 = probs[:, 0]  # class index 0 as in inference script
            scores = [softmax_to_score(p.item()) for p in p1]
            all_preds.extend(zip(ids, scores))

    # Calculate accuracy within standard deviation
    acc = accuracy_within_standard_deviation(all_preds, gold_data)

    # Calculate Spearman correlation (matching evaluate.py logic)
    gold_list = []
    pred_list = []
    for pid, pred in all_preds:
        gold_avg = sum(gold_data[str(pid)]["choices"]) / len(gold_data[str(pid)]["choices"])
        gold_list.append(gold_avg)
        pred_list.append(pred)

    spearman_corr, _ = spearmanr(pred_list, gold_list)

    return acc, spearman_corr, all_preds


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = json.load(Path("data/train.json").open())
    dev_data = json.load(Path("data/dev.json").open())

    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_ds = GlossDataset(train_data)
    dev_ds = GlossDataset(dev_data)

    batch_size = 8
    epochs = 5
    lr = 2e-5
    weight_decay = 0.01

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )
    # Separate loader for evaluation (no shuffle)
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Track metrics for plotting
    train_acc_history = []
    train_spearman_history = []
    dev_acc_history = []
    dev_spearman_history = []

    best_acc = -1.0
    best_preds = None
    best_state = None

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, (_ids, inputs, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[:, 0]
            loss = F.binary_cross_entropy(probs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            if step % 50 == 0 or step == len(train_loader):
                avg_loss = epoch_loss / step
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Avg train loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)

        # Evaluate on both train and dev sets
        train_acc, train_spearman, _train_preds = evaluate(model, train_eval_loader, train_data, device)
        dev_acc, dev_spearman, dev_preds = evaluate(model, dev_loader, dev_data, device)

        # Record metrics
        train_acc_history.append(train_acc)
        train_spearman_history.append(train_spearman)
        dev_acc_history.append(dev_acc)
        dev_spearman_history.append(dev_spearman)

        print(f"Epoch {epoch+1} done. Train loss: {avg_epoch_loss:.4f}")
        print(f"  Train - Acc: {train_acc:.4f}, Spearman: {train_spearman:.4f}")
        print(f"  Dev   - Acc: {dev_acc:.4f}, Spearman: {dev_spearman:.4f}")

        # Track best model based on dev accuracy
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_preds = dev_preds
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  New best dev acc: {best_acc:.4f}")

    # Save best predictions to file (JSONL format, requested filename)
    output_path = Path("trained_gloss_roberta__predictions_dev_Sent-CLS.jsonl")
    with output_path.open("w") as f:
        for pid, pred in best_preds:
            f.write(json.dumps({"id": str(pid), "prediction": int(pred)}) + "\n")

    # Plot results
    epochs_list = list(range(1, epochs + 1))

    # Plot 1: Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_acc_history, marker="o", color="orange", label="Train")
    plt.plot(epochs_list, dev_acc_history, marker="s", color="blue", label="Dev")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Gloss-RoBERTa (Sent-CLS) — Accuracy vs Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch_Sent-CLS.png", dpi=150)
    plt.close()
    print(f"\nSaved accuracy plot to: accuracy_vs_epoch_Sent-CLS.png")

    # Plot 2: Spearman Correlation vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_spearman_history, marker="o", color="orange", label="Train")
    plt.plot(epochs_list, dev_spearman_history, marker="s", color="blue", label="Dev")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Spearman Correlation", fontsize=12)
    plt.title("Gloss-RoBERTa (Sent-CLS) — Spearman vs Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("spearman_vs_epoch_Sent-CLS.png", dpi=150)
    plt.close()
    print(f"Saved Spearman plot to: spearman_vs_epoch_Sent-CLS.png")

    print(f"\nBest dev accuracy: {best_acc:.4f}")

    # Optionally save best model weights (commented out to keep minimal side-effects)
    # torch.save(best_state, "best_gloss_roberta_sent_cls.pt")


if __name__ == "__main__":
    # Do not auto-run training when imported.
    train()
