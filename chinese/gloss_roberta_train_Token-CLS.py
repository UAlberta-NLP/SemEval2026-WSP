import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# ---------- Preprocessing (Token-CLS variant) ----------
def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_context(sample: Dict) -> Tuple[str, int, int]:
    pre = normalize_text(sample.get("precontext", ""))
    sentence = normalize_text(sample.get("sentence", ""))
    ending = normalize_text(sample.get("ending", ""))

    parts = [p for p in [pre, sentence, ending] if p]
    context = " ".join(parts)

    if sentence:
        sentence_start = len(pre) + 1 if pre else 0
        sentence_end = sentence_start + len(sentence)
    else:
        sentence_start = -1
        sentence_end = -1

    return context, sentence_start, sentence_end


def build_gloss(sample: Dict) -> str:
    # Token-CLS variant does not prepend the target word to the gloss.
    return sample["judged_meaning"]


def softmax_to_score(prob_first_class: float) -> int:
    score = round(1 + 4 * prob_first_class)
    return max(1, min(5, int(score)))


def find_target_span(context: str, target: str) -> Tuple[int, int]:
    """
    Locate the first occurrence of the target word in the context using
    case-insensitive word boundaries. Returns (start, end) character offsets.
    """
    pattern = re.compile(rf"\\b{re.escape(target)}\\b", flags=re.IGNORECASE)
    match = pattern.search(context)
    return match.span() if match else (-1, -1)


def target_token_indices(context: str, target: str, encoding, sentence_start: int, sentence_end: int) -> List[int]:
    """
    Map the target span in the *sentence portion* of the context to token indices
    in the joint context-gloss encoding. Falls back to the first context token
    if the target cannot be matched (should be rare).
    """
    if sentence_start < 0 or sentence_end < 0:
        start, end = -1, -1
    else:
        sentence_text = context[sentence_start:sentence_end]
        rel_start, rel_end = find_target_span(sentence_text, target)
        start = sentence_start + rel_start if rel_start >= 0 else -1
        end = sentence_start + rel_end if rel_end >= 0 else -1

    seq_ids_obj = getattr(encoding, "sequence_ids", None)
    seq_ids = seq_ids_obj() if callable(seq_ids_obj) else seq_ids_obj
    offsets = getattr(encoding, "offsets", encoding.offsets)

    indices: List[int] = []
    for idx, (off_start, off_end) in enumerate(offsets):
        if seq_ids[idx] != 0:
            continue  # only context tokens
        if off_start == off_end == 0:
            continue  # special tokens
        if off_end > start and off_start < end:
            indices.append(idx)
    if not indices:
        # Fallback to the first non-special context token
        for idx, seq_id in enumerate(seq_ids):
            if seq_id == 0 and offsets[idx] != (0, 0):
                indices = [idx]
                break
    return indices if indices else [0]


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
        context, sentence_start, sentence_end = build_context(sample)
        gloss = build_gloss(sample)
        # Soft label mapped to [0,1]
        target_prob = (float(sample["average"]) - 1.0) / 4.0
        return sid, context, gloss, target_prob, sample["homonym"], (sentence_start, sentence_end)


def collate_fn(batch, tokenizer, device):
    ids, contexts, glosses, targets, target_words, sentence_spans = zip(*batch)
    enc = tokenizer(
        list(contexts),
        list(glosses),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    target_positions = [
        target_token_indices(
            contexts[i], target_words[i], enc.encodings[i], sentence_spans[i][0], sentence_spans[i][1]
        )
        for i in range(len(contexts))
    ]
    # Remove offset mapping before moving tensors to device
    enc.pop("offset_mapping")
    return (
        list(ids),
        {k: v.to(device) for k, v in enc.items()},
        torch.tensor(targets, dtype=torch.float, device=device),
        target_positions,
    )


# ---------- Model (Token-CLS pooling) ----------
class TokenCLSGlossModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: Dict[str, torch.Tensor], target_positions: List[List[int]]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled_targets = []
        for i, positions in enumerate(target_positions):
            index_tensor = torch.tensor(positions, device=hidden_states.device)
            token_vecs = hidden_states[i, index_tensor, :]
            pooled = token_vecs.mean(dim=0)
            pooled_targets.append(pooled)
        pooled_targets = torch.stack(pooled_targets, dim=0)
        pooled_targets = self.dropout(pooled_targets)
        logits = self.classifier(pooled_targets)
        return logits


# ---------- Training / Evaluation ----------
def evaluate(model, dataloader, gold_data, device) -> Tuple[float, float, List[Tuple[str, int]]]:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for ids, inputs, _targets, target_positions in dataloader:
            logits = model(inputs, target_positions)
            probs = F.softmax(logits, dim=-1)
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
    model = TokenCLSGlossModel(model_name)
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

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "weight_decay": weight_decay},
            {"params": model.classifier.parameters(), "weight_decay": weight_decay},
        ],
        lr=lr,
    )
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

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, (_ids, inputs, targets, target_positions) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logits = model(inputs, target_positions)
            probs = F.softmax(logits, dim=-1)[:, 0]
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
            print(f"  New best dev acc: {best_acc:.4f}")

    # Save best predictions to file (JSONL format, requested filename)
    output_path = Path("trained_gloss_roberta__predictions_dev_Token-CLS.jsonl")
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
    plt.title("Gloss-RoBERTa (Token-CLS) — Accuracy vs Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch_Token-CLS.png", dpi=150)
    plt.close()
    print(f"\\nSaved accuracy plot to: accuracy_vs_epoch_Token-CLS.png")

    # Plot 2: Spearman Correlation vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_spearman_history, marker="o", color="orange", label="Train")
    plt.plot(epochs_list, dev_spearman_history, marker="s", color="blue", label="Dev")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Spearman Correlation", fontsize=12)
    plt.title("Gloss-RoBERTa (Token-CLS) — Spearman vs Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("spearman_vs_epoch_Token-CLS.png", dpi=150)
    plt.close()
    print(f"Saved Spearman plot to: spearman_vs_epoch_Token-CLS.png")

    print(f"\\nBest dev accuracy: {best_acc:.4f}")

    # Optionally save best model weights (commented out to keep minimal side-effects)
    # torch.save(model.state_dict(), \"best_gloss_roberta_token_cls.pt\")


if __name__ == "__main__":
    # Do not auto-run training when imported.
    train()
