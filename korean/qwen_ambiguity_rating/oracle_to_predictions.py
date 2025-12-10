#!/usr/bin/env python3
"""
oracle_to_predictions.py
Convert qwen_oracle outputs to prediction JSONL format.

Logic:
- CLEAR: the clear meaning gets 5, the other gets clear_other_rating (2 or 3)
- AMBIGUOUS: both meanings get ambiguous_rating (2, 3, or 4)

Rating scale:
1 = Very low plausibility (not used by oracle)
2 = Low plausibility
3 = Moderate plausibility
4 = High plausibility
5 = Very high / clearly intended
"""

import argparse, json, os, glob, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_dir", default="outputs_oracle", help="Folder containing oracle <key>.json files")
    ap.add_argument("--data", required=True, help="Path to original data JSON (e.g., data/dev.json)")
    ap.add_argument("--out", default="qwen_oracle_predictions.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    # Load original data to map IDs to their judged_meaning
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Build mapping: id -> judged_meaning
    id_to_meaning = {k: v["judged_meaning"] for k, v in data.items()}

    # Load oracle files
    files = sorted(glob.glob(os.path.join(args.oracle_dir, "*.json")),
                   key=lambda p: (len(os.path.basename(p).split('.')[0]),
                                  os.path.basename(p).split('.')[0]))
    if not files:
        print(f"No JSON files found in {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    # Process oracle outputs and collect predictions
    predictions = {}
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as rf:
                oracle = json.load(rf)
            
            original_keys = oracle.get("original_keys", [])
            judgment = oracle.get("judgment", "AMBIGUOUS")
            clear_meaning_index = oracle.get("clear_meaning_index")  # 1-based
            clear_other_rating = oracle.get("clear_other_rating", 2)  # Rating for non-intended meaning
            ambiguous_rating = oracle.get("ambiguous_rating", 3)  # Rating when ambiguous
            meanings_considered = oracle.get("meanings_considered", [])
            
            if not original_keys:
                print(f"[skip] {fpath}: no original_keys", file=sys.stderr)
                continue
            
            if judgment == "AMBIGUOUS":
                # Both get the ambiguous_rating (2, 3, or 4)
                rating = ambiguous_rating if ambiguous_rating in [2, 3, 4] else 3
                for key in original_keys:
                    predictions[key] = rating
            elif judgment == "CLEAR" and clear_meaning_index is not None:
                # The clear meaning gets 5, the other gets clear_other_rating (2 or 3)
                clear_meaning_text = meanings_considered[clear_meaning_index - 1] if clear_meaning_index <= len(meanings_considered) else None
                other_rating = clear_other_rating if clear_other_rating in [2, 3] else 2
                
                for key in original_keys:
                    key_meaning = id_to_meaning.get(key)
                    if key_meaning == clear_meaning_text:
                        predictions[key] = 4  # Clearly intended meaning
                    else:
                        predictions[key] = other_rating  # Other meaning's plausibility
            else:
                # Fallback: treat as ambiguous with moderate plausibility
                for key in original_keys:
                    predictions[key] = 3
                    
        except Exception as e:
            print(f"[skip] {fpath}: {e}", file=sys.stderr)

    # Sort keys numerically and write output
    sorted_keys = sorted(predictions.keys(), key=lambda x: (len(x), x))
    
    with open(args.out, "w", encoding="utf-8") as w:
        for key in sorted_keys:
            line = {"id": key, "prediction": predictions[key]}
            w.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(predictions)} predictions to {args.out}")
    
    # Print summary stats
    counts = {}
    for v in predictions.values():
        counts[v] = counts.get(v, 0) + 1
    stats = ", ".join(f"{k}={counts.get(k,0)}" for k in sorted(counts.keys()))
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()
