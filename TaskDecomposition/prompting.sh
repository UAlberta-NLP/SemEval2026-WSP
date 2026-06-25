#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if the user provided an input file and an output directory
if [ "$#" -ne 2 ]; then
  echo "Error: Missing arguments."
  echo "Usage: ./prompting.sh <path_to_data_file> <output_directory>"
  echo "Example: ./prompting.sh ../data/train.json outputs/train"
  exit 1
fi

DATA_FILE=$1
OUT_DIR=$2

# Ensure the targeted output directory exists
mkdir -p "$OUT_DIR"

echo "Starting GPT Prompting Pipeline on: $DATA_FILE"
echo "Saving outputs to: $OUT_DIR"
echo "---------------------------------------------------"

echo "1/5: Running gpt_ambiguous.py..."
python prompts/gpt_ambiguous.py --in "$DATA_FILE" --out_json "$OUT_DIR"

echo "2/5: Running gpt_neither_fits.py..."
python prompts/gpt_neither_fits.py --in "$DATA_FILE" --out_json "$OUT_DIR"

echo "3/5: Running gpt_preference.py..."
python prompts/gpt_preference.py --in "$DATA_FILE" --out_json "$OUT_DIR"

echo "4/5: Running gpt_correct.py..."
python prompts/gpt_correct.py --in "$DATA_FILE" --out_json "$OUT_DIR"

echo "5/5: Running gpt_rating.py..."
python prompts/gpt_rating.py --in "$DATA_FILE" --out_json "$OUT_DIR"

echo "---------------------------------------------------"
echo "Pipeline complete! Results saved to the '$OUT_DIR' directory."