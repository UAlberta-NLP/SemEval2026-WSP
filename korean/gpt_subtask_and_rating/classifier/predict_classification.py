#!/usr/bin/env python3
import json
import argparse
import pandas as pd
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


# ---------------------------------------------------------------
# Load GPT ambiguity flags (CLEAR/AMBIGUOUS per pair)
# ---------------------------------------------------------------
def load_gpt_ambiguity(path: Path):
    ambig = {}
    data = json.load(path.open("r", encoding="utf-8"))
    for r in data:
        k1, k2 = r["key1"], r["key2"]
        is_amb = (r["model_choice"] == "AMBIGUOUS")
        ambig[k1] = 1 if is_amb else 0
        ambig[k2] = 1 if is_amb else 0
    return ambig


# ---------------------------------------------------------------
# Binary one-hot for GPT 1/2 decisions
# ---------------------------------------------------------------
def load_gpt_choice(path: Path):
    score = {}
    data = json.load(path.open("r", encoding="utf-8"))
    for r in data:
        k1, k2 = r["key1"], r["key2"]
        c = r["model_choice"]
        score.setdefault(k1, 0.0)
        score.setdefault(k2, 0.0)
        if c == 1:
            score[k1] = 1
            score[k2] = 0
        elif c == 2:
            score[k1] = 0
            score[k2] = 1
    return score


# ---------------------------------------------------------------
# NEW: Loader for GPT "NEITHER plausible"
# ---------------------------------------------------------------
def load_gpt_neither(path: Path):
    feat = {}
    data = json.load(path.open("r", encoding="utf-8"))

    for r in data:
        k1, k2 = r["key1"], r["key2"]

        is_neither = (r.get("model_choice") == "NEITHER" or r.get("neither_fits") == 1)

        val = 1 if is_neither else 0
        feat[k1] = val
        feat[k2] = val

    return feat


# ---------------------------------------------------------------
# >>> NEW: Loader for GPT "guaranteed correct"
# ---------------------------------------------------------------
def load_gpt_correct(path: Path):
    feat = {}
    data = json.load(path.open("r", encoding="utf-8"))

    # Structure: [{"id": "...", "model_choice": "CORRECT", ...}, ...]
    for r in data:
        sid = r["id"]
        is_correct = (r.get("model_choice") == "CORRECT")
        feat[sid] = 1 if is_correct else 0

    return feat


# ---------------------------------------------------------------
# >>> NEW: Loader for GPT rating (1-5 plausibility)
# ---------------------------------------------------------------
def load_gpt_rating(path: Path):
    """Loads item-level GPT plausibility ratings (1-5) from gpt_rating.py output"""
    feat = {}

    data = json.load(path.open("r", encoding="utf-8"))
    for r in data:
        sid = r["id"]
        score = r.get("model_score")
        feat[sid] = score if score is not None else 3  # default to 3 (ambiguous) if missing

    return feat


# ---------------------------------------------------------------
# Acc within SD OR 1.0
# ---------------------------------------------------------------
def acc_within_sd_or_one(preds, labels, sds):
    preds = np.array(preds)
    labels = np.array(labels)
    sds = np.array(sds)
    thresholds = np.maximum(sds, 1.0)
    return np.mean(np.abs(preds - labels) < thresholds)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args):
    dev_json = Path(args.dev_json)
    gpt_json = Path(args.gpt_preference_json)
    ambig_json = Path(args.ambig_json)
    neither_json = Path(args.neither_json)
    correct_json = Path(args.correct_json) 
    rating_json = Path(args.rating_json)    
    model_path = Path(args.model)
    out_csv = Path(args.out)

    # ============================
    # Load dev data
    # ============================
    with dev_json.open("r", encoding="utf-8") as f:
        dev_data = json.load(f)

    rows = []

    # GPT-based features
    gpt_choice = load_gpt_choice(gpt_json)
    gpt_ambig = load_gpt_ambiguity(ambig_json)
    gpt_neither = load_gpt_neither(neither_json)
    gpt_correct = load_gpt_correct(correct_json)   
    gpt_rating = load_gpt_rating(rating_json)      

    # ============================
    # Pair reconstruction
    # ============================
    ctx_groups = {}
    for sid, entry in dev_data.items():
        key = (entry["precontext"], entry["sentence"], entry["ending"])
        ctx_groups.setdefault(key, []).append(sid)

    records = []

    for key, ids in ctx_groups.items():
        if len(ids) != 2:
            continue

        i, j = ids

        def make_record(sid, entry):
            try:
                label_thing = float(entry["average"])
                stdev_thing = float(entry["stddev"])
            except (ValueError, TypeError):
                label_thing = '???'
                stdev_thing = '???'
            return {
                "id": sid,
                "label": label_thing,
                
                "sd": stdev_thing,

                # GPT features
                "gpt_choice": gpt_choice.get(sid, 0),
                "gpt_ambiguous": gpt_ambig.get(sid, 0),
                "gpt_neither": gpt_neither.get(sid, 0),
                "gpt_correct": gpt_correct.get(sid, 0),   # >>> NEW
                "gpt_rating": gpt_rating.get(sid, 3),     # >>> NEW: GPT rating (1-5)

                "has_ending": 1 if entry.get("ending", "") != "" else 0,
            }

        records.append(make_record(i, dev_data[i]))
        records.append(make_record(j, dev_data[j]))

    df = pd.DataFrame(records)
    print("Dev samples:", len(df))

    # ============================
    # Predict
    # ============================
    model = joblib.load(model_path)

    feature_cols = [c for c in df.columns if c not in ("id", "label", "sd")]
    X = df[feature_cols]
    y = df["label"]  # Original average for metrics
    sd = df["sd"]
    
    # Classifier outputs 0-4, shift back to 1-5
    preds = model.predict(X) + 1

    # ============================
    # Metrics
    # ============================
    try:
        mse = mean_squared_error(y, preds)
        spearman, _ = spearmanr(y, preds)
        acc_sd1 = acc_within_sd_or_one(preds, y, sd)
        
        print("\n=== DEV METRICS (XGBClassifier + GPT features) ===")
        print(f"MSE:                 {mse:.4f}")
        print(f"Spearman ρ:          {spearman:.4f}")
        print(f"Acc within SD or 1:  {acc_sd1:.4f}")
    except ValueError:
        mse = 'undefined'
        spearman = 'undefined'
        acc_sd1 = 'undefined'

    

    # ============================
    # Save predictions
    # ============================
    out_df = pd.DataFrame({
        "id": df["id"],
        "gold": y,
        "pred": preds,
        "sd": sd,
    })
    out_df.to_csv(out_csv, index=False)
    print("Saved predictions →", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_json", required=True)
    parser.add_argument("--gpt_preference_json", required=True)
    parser.add_argument("--ambig_json", required=True)
    parser.add_argument("--neither_json", required=True)
    parser.add_argument("--correct_json", required=True)     # >>> NEW
    parser.add_argument("--rating_json", required=True)      # >>> NEW: GPT rating
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
