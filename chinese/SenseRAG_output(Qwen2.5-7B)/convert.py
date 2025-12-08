#!/usr/bin/env python3
import csv
import json
import sys

def tsv_to_jsonl(input_file, output_file):
    """
    Read a TSV file and extract only the 'idx' and 'final_score' columns.
    Each row becomes one JSON object written as a JSONL line.
    """
    with open(input_file, "r", encoding="utf-8", newline="") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        # Read TSV using DictReader for column access by header name
        reader = csv.DictReader(fin, delimiter="\t")

        # Ensure required fields exist
        if "idx" not in reader.fieldnames or "final_score" not in reader.fieldnames:
            raise ValueError(
                f"Required columns 'idx' or 'final_score' not found. Columns found: {reader.fieldnames}"
            )

        # Process each row
        for row in reader:

            # Convert idx to int if possible
            try:
                idx = str(row["idx"])
            except ValueError:
                idx = row["idx"]  # keep as string if conversion fails

            # Convert final_score to float if possible
            try:
                final_score = float(row["final_score"])
            except ValueError:
                final_score = row["final_score"]

            # Build JSON object for this line
            obj = {
                "id": idx,
                "prediction": final_score
            }

            # Write JSON object as one JSONL line
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done: {input_file} -> {output_file}")


if __name__ == "__main__":
    # Require exactly two arguments: input TSV and output JSONL
    if len(sys.argv) != 3:
        print("Usage: python tsv_to_jsonl.py <input.tsv> <output.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    tsv_to_jsonl(input_path, output_path)
