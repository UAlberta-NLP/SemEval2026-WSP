# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is UAlberta's submission for SemEval-2026 Task 5: Rating Word Senses Plausibility (WSP) in Ambiguous Sentences through Narrative Understanding. The task involves predicting plausibility ratings (1-5 scale) for ambiguous word meanings in narrative contexts.

## Repository Structure

The repository follows a multi-team collaborative structure:

- **`src/`** — Shared evaluation scripts and baseline implementations
  - `src/eval/scoring.py` — Main evaluation script
  - `src/eval/format_check.py` — Prediction format validation
  - `src/baselines/` — Reference baseline models
- **`res/`** — Shared resources (data, predictions, scores)
  - `res/data/` — Train/dev datasets
  - `res/predictions/` — Output prediction files
  - `res/scores/` — Evaluation results
- **Team directories** — Language-specific team workspaces (isolated):
  - `chinese/` — Chinese team implementations
  - `korean/` — Korean team implementations (GPT-based classifiers, Qwen models)
  - `persian/` — Persian team implementations
  - `urdu/` — Urdu team implementations (GlossDeBERTa, OHPT approaches)

### Multi-Team Workflow
Each team works exclusively in their designated directory to avoid merge conflicts. Teams share common data and evaluation scripts from `src/` and `res/` but maintain separate model implementations.

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
**CRITICAL**: Baseline scripts use hardcoded relative paths (`"data/" + SET + ".json"`) and MUST be run from the `res/` directory. This is a known limitation of the baseline implementations.

When creating new models:
- Follow the same pattern and run from `res/`, or
- Use absolute paths or proper path resolution with `os.path.join()` and `__file__`

### Dependencies
- **Root level**: scipy (for spearmanr)
- **Team-specific**: Each team directory may have its own `requirements.txt` for specialized dependencies (transformers, torch, xgboost, etc.)
- Standard library: json, statistics, sys, os, random

## Working with Team Implementations

### Korean Team
Located in `korean/`. Key approaches:
- **GPT Subtask + Rating Classifier**: Prompts GPT-4o to extract features (ambiguity, preference, correctness), then trains XGBoost classifier/regressor
  - Run prompts: `python prompts/gpt_*.py --in <DATA> --out_json outputs`
  - Train: `python classifier/train_classification.py --train_json <DATA> --gpt_preference_json outputs/train/train_preference.json ...`
  - Predict: `python classifier/predict_classification.py --dev_json <DATA> ... --model xgb.pkl --out predictions.csv`
- **Qwen Ambiguity Rating**: Uses Qwen model to judge clear vs ambiguous meanings
  - Run: `python qwen_oracle.py --data_json <DATA> --out_dir outputs_oracle`

### Urdu Team
Located in `urdu/`. Key approaches:
- **GlossDeBERTa**: Fine-tuned DeBERTa model on gloss-context pairs
  - Training: `python GlossDeBERTa/run_classifier_WSD_token.py`
  - Inference: `python GlossDeBERTa/predict_score.py`
- **OHPT (One Hot Pool Tagging)**: BabelNet-based approach with translation

### Chinese Team
Located in `chinese/`. Key approaches:
- **SenseRAG Pipeline**: RAG-based sense disambiguation with Mistral/Qwen2.5
- **Few-shot prompting**: Direct LLM prompting with examples
- **GlossRoBERTa**: Fine-tuned RoBERTa variants (Sent-CLS, Token-CLS)

## Important Notes

### Data Location
All teams access shared data from `res/data/`:
- `res/data/train.json` — Training set
- `res/data/dev.json` — Development set
- `res/data/dev_solution.jsonl` — Dev set labels (auto-generated by baselines.sh)

Test set is not yet released. Always evaluate on dev set during development.

### Creating New Models
When implementing new approaches:
1. Work in your team directory (e.g., `chinese/my_new_model.py`)
2. Read data using paths relative to your script location or absolute paths
3. Output predictions to `res/predictions/<model_name>_predictions_dev.jsonl`
4. Use the shared evaluation script: `python3 src/eval/scoring.py res/data/dev_solution.jsonl res/predictions/<model_name>_predictions_dev.jsonl res/scores/<model_name>_scores.json`

### Submission
Submission is via CodaBench: https://www.codabench.org/competitions/10877/
- Zip your `predictions.jsonl` file
- Upload to "My Submissions" tab
