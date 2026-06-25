#!/bin/bash

# Exit immediately if a command fails
set -e

# Ensure the user provides exactly 3 arguments
if [ "$#" -ne 3 ]; then
  echo "Error: Missing arguments."
  echo "Usage: ./run_regression.sh <TRAIN_DATA_FILE> <INFER_DATA_FILE> <OUTPUT_JSONL_NAME>"
  exit 1
fi

TRAIN_FILE=$1
INFER_FILE=$2
OUT_JSONL=$3

echo "Starting Regression Pipeline..."
echo "---------------------------------------------------"

echo "1/3: Training XGBoost Regression model..."
python classifier/train_regression.py \
  --train_json "$TRAIN_FILE" \
  --gpt_preference_json outputs/train/train_preference.json \
  --ambig_json outputs/train/train_ambiguity.json \
  --neither_json outputs/train/train_neither.json \
  --correct_json outputs/train/train_correct.json \
  --rating_json outputs/train/train_rating.json

echo "2/3: Predicting on inference set..."
python classifier/predict_regression.py \
  --dev_json "$INFER_FILE" \
  --gpt_preference_json outputs/infer/infer_preference.json \
  --ambig_json outputs/infer/infer_ambiguity.json \
  --neither_json outputs/infer/infer_neither.json \
  --correct_json outputs/infer/infer_correct.json \
  --rating_json outputs/infer/infer_rating.json \
  --model xgb_regression.pkl \
  --out infer_predictions_regression.csv

echo "3/3: Converting CSV predictions to JSONL format..."
python tools/csv_to_jsonl.py \
  --in_csv infer_predictions_regression.csv \
  --out_jsonl "$OUT_JSONL"

echo "---------------------------------------------------"
echo "Regression complete! Final predictions saved to: $OUT_JSONL"