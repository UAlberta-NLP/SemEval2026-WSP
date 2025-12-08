import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


# ---------- Preprocessing (matches gloss_roberta_inference.py) ----------
import re


def highlight_target(context: str, target: str) -> str:
    pattern = re.compile(rf"\\b{re.escape(target)}\\b", flags=re.IGNORECASE)

    def replacer(match: re.Match) -> str:
        word = match.group(0)
        if word.startswith('"') and word.endswith('"'):
            return word
        return f'"{word}"'

    return pattern.sub(replacer, context)


def build_context(sample: Dict) -> str:
    parts = [sample.get("precontext", ""), sample.get("sentence", ""), sample.get("ending", "")]
    return " ".join(" ".join(parts).split())


def build_gloss(sample: Dict) -> str:
    return f"{sample['homonym']}: {sample['judged_meaning']}"


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
        context = highlight_target(build_context(sample), sample["homonym"])
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
def evaluate(model, dataloader, gold_data, device) -> Tuple[float, List[Tuple[str, int]]]:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for ids, inputs, _targets in dataloader:
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            p1 = probs[:, 0]  # class index 0 as in inference script
            scores = [softmax_to_score(p.item()) for p in p1]
            all_preds.extend(zip(ids, scores))
    acc = accuracy_within_standard_deviation(all_preds, gold_data)
    return acc, all_preds


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
    patience = 2  # stop if no dev acc improvement for this many eval steps

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    best_acc = -1.0
    best_preds = None
    epochs_no_improve = 0

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, (ids, inputs, targets) in enumerate(train_loader, start=1):
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
        dev_acc, dev_preds = evaluate(model, dev_loader, dev_data, device)
        print(f"Epoch {epoch+1} done. Train loss: {avg_epoch_loss:.4f} | Dev acc (within SD): {dev_acc:.4f}")
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_preds = dev_preds
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"New best dev acc: {best_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Save best predictions to file (JSONL format, requested filename)
    output_path = Path("trained_gloss_roberta__predictions_dev.jsonl")
    with output_path.open("w") as f:
        for pid, pred in best_preds:
            f.write(json.dumps({"id": str(pid), "prediction": int(pred)}) + "\n")

    # Optionally save best model weights (commented out to keep minimal side-effects)
    # torch.save(best_state, "best_gloss_roberta.pt")


if __name__ == "__main__":
    # Do not auto-run training when imported.
    train()
