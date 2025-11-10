#!/bin/bash

# Script to run all baseline models and evaluate them
# Usage: bash baselines.sh

set -e  # Exit on error

echo "================================================================================"
echo "RANDOM BASELINE"
echo "================================================================================"
cd res
python3 ../src/baselines/random_baseline.py
cd ..
python3 src/eval/scoring.py res/data/dev_solution.jsonl res/predictions/random_predictions_dev.jsonl res/scores/random_scores.json

echo ""
echo "================================================================================"
echo "MAJORITY BASELINE"
echo "================================================================================"
cd res
python3 ../src/baselines/majority_baseline.py
cd ..
python3 src/eval/scoring.py res/data/dev_solution.jsonl res/predictions/majority_predictions_dev.jsonl res/scores/majority_scores.json

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "Random Baseline:"
python3 -c "import json, math; scores = json.load(open('res/scores/random_scores.json')); spearman = 'N/A' if scores['spearman'] is None or math.isnan(scores['spearman']) else f\"{scores['spearman']:.4f}\"; print(f\"  Spearman: {spearman}\"); print(f\"  Accuracy: {scores['accuracy']:.4f}\")"
echo ""
echo "Majority Baseline:"
python3 -c "import json, math; scores = json.load(open('res/scores/majority_scores.json')); spearman = 'N/A' if scores['spearman'] is None or math.isnan(scores['spearman']) else f\"{scores['spearman']:.4f}\"; print(f\"  Spearman: {spearman}\"); print(f\"  Accuracy: {scores['accuracy']:.4f}\")"
echo ""
echo "Results saved to: res/scores/random_scores.json and res/scores/majority_scores.json"
echo "================================================================================"
