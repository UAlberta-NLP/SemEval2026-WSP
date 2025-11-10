# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is UAlberta's submission for SemEval-2026 Task 5: Rating Word Senses Plausibility (WSP) in Ambiguous Sentences through Narrative Understanding. The task involves predicting plausibility ratings (1-5 scale) for ambiguous word meanings in narrative contexts.

## Data Structure

### Input Data Format (train.json, dev.json)
JSON files with nested dictionaries where keys are sample IDs:
- `homonym`: The ambiguous word
- `judged_meaning`: The specific word sense being evaluated
- `precontext`: Context before the target sentence
- `sentence`: The sentence containing the ambiguous word
- `ending`: Context after the target sentence
- `example_sentence`: Example usage of the judged meaning
- `choices`: List of 5 human ratings (1-5)
- `average`: Mean of human ratings
- `stdev`: Standard deviation of ratings
- `nonsensical`: Per-annotator flags
- `sample_id`: Original sample identifier

### Prediction Format
JSONL format (one JSON object per line):
```json
{"id": "0", "prediction": 4}
```
Each prediction must be an integer 1-5 for every sample ID in the test set.

### Gold Label Format (solution.jsonl)
JSONL format with all human judgments:
```json
{"id": "0", "label": [4, 5, 3, 1, 5]}
```

## Evaluation Metrics

The scoring system uses two metrics:

1. **Spearman Correlation**: Measures correlation between predictions and average human ratings
2. **Accuracy within Standard Deviation**: Proportion of predictions within either:
   - ±1 standard deviation of the human average, OR
   - Distance < 1 from the human average

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Commands

### Run All Baselines
Run both baselines and evaluate them in one command:
```bash
bash baselines.sh
```
This will:
- Generate random baseline predictions
- Generate majority baseline predictions
- Evaluate both against test solution
- Save results to `random_scores.json` and `majority_scores.json`
- Print a summary of results

### Run Individual Baselines
Baselines must be run from the `res/` directory due to hardcoded relative paths:
```bash
cd res
python3 ../src/baselines/random_baseline.py
python3 ../src/baselines/majority_baseline.py
```

### Evaluate Predictions
```bash
python3 src/eval/scoring.py <solution_file.jsonl> <predictions_file.jsonl> <output_scores.json>
```
Example:
```bash
python3 src/eval/scoring.py res/data/dev_solution.jsonl res/predictions/my_predictions.jsonl my_scores.json
```

This will:
- Validate prediction format
- Print Spearman correlation and accuracy scores
- Save results to the specified JSON file

**Note:** The test set is not yet released. Evaluate on dev set using `res/data/dev_solution.jsonl` (generated automatically by `baselines.sh`).

### Format Validation Only
```bash
python3 src/eval/format_check.py <predictions_file.jsonl>
```

## Architecture Notes

- **src/eval/scoring.py**: Main evaluation script that imports format_check and computes both metrics
- **src/eval/format_check.py**: Validates prediction files (proper JSON, correct IDs, predictions in 1-5 range)
- **src/baselines/**: Simple baseline implementations
  - `random_baseline.py`: Randomly samples from [1,2,3,4,5]
  - `majority_baseline.py`: Predicts majority label (4) for all samples

### Path Dependencies
Baseline scripts use hardcoded relative paths (`"data/" + SET + ".json"`) and expect to be run from the `res/` directory. When creating new models, either:
- Follow the same pattern and run from `res/`, or
- Use absolute paths or proper path resolution

### Dependencies
- scipy (for spearmanr)
- Standard library: json, statistics, sys, os, random
