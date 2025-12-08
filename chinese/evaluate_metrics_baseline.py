import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# =========================
# Config
# =========================
OUTPUT_TSV = "baseline_output.tsv"
DEV_JSON = "dev.json"

# =========================
# 1) Load prediction file
# =========================
# Assumes a header row and tab-separated columns.
df = pd.read_csv(OUTPUT_TSV, sep="\t")

# Build a lookup: sample index/id -> predicted final score
# If your key is 'sample_id' instead of 'idx', change below accordingly.
pred_dict = dict(zip(df["idx"], df["final_score"]))

# =========================
# 2) Load gold (dev) file
# =========================
with open(DEV_JSON, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

gold_scores = []
pred_scores = []

# gold_data is assumed to be a dict: { "<id>": { ... "average": <float> ... }, ... }
# If it's a list, adapt iteration accordingly.
for sid, sample in gold_data.items():
    gold = sample.get("average", None)
    pred = pred_dict.get(int(sid), None)  # NOTE: cast sid to int to match df["idx"]
    if gold is not None and pred is not None:
        gold_scores.append(float(gold))
        pred_scores.append(float(pred))

gold_scores = np.array(gold_scores, dtype=float)
pred_scores = np.array(pred_scores, dtype=float)

if len(gold_scores) == 0:
    raise ValueError("No matched samples between dev.json and output.tsv. "
                     "Check that the keys/ids align (e.g., 'idx' vs 'sample_id').")

# =========================
# 3) Spearman rank correlation (ordering agreement)
# =========================
spearman_corr, _ = spearmanr(gold_scores, pred_scores)

# =========================
# 4) Error metrics (magnitude of deviations)
# =========================
abs_err = np.abs(gold_scores - pred_scores)
mae = abs_err.mean()                           # Mean Absolute Error
rmse = np.sqrt(np.mean((gold_scores - pred_scores) ** 2))  # Root Mean Squared Error

# =========================
# 5) Accuracy within one gold standard deviation
# =========================
std_dev = float(np.std(gold_scores))
within_std = int(np.sum(abs_err <= std_dev))
acc_within_std = within_std / len(gold_scores)

# =========================
# 6) Print results
# =========================
print("📊 Evaluation Metrics")
print("---------------------------")
print(f"Number of samples: {len(gold_scores)}")
print(f"Gold std deviation: {std_dev:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Accuracy within std: {acc_within_std:.4f} ({within_std}/{len(gold_scores)})")
