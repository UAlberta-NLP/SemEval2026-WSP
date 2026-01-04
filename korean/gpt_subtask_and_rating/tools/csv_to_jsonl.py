#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True, help="Input predictions CSV")
    parser.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_jsonl)

    with in_path.open("r", encoding="utf-8", newline="") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            # Adjust column name if needed: "pred" is what we wrote in predict.py
            rec = {
                "id": str(row["id"]),
                "prediction": float(row["pred"]),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL to {out_path}")

if __name__ == "__main__":
    main()
