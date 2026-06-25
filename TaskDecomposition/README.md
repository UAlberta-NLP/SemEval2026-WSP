# Task Decomposition Pipeline


### Environment Setup
First, install the necessary Python dependencies:

```bash
python -m pip install -r requirements.txt
```

We first collect subtask features by prompting GPT-5 on the train and dev sets. Run these scripts on both `train.json` and `dev.json` (or your preferred inference file).

### Prompting With Bash Script (`prompting.sh`)

**Running the Prompting Pipeline**

To run the full task decomposition and rating pipeline, execute the bash script, passing your data file as the argument:

```bash
# Make the script executable (only needed once)
chmod +x prompting.sh

# Run the pipeline on the training set
./prompting.sh ../data/train.json outputs/train

# Run the pipeline on the inference set
./prompting.sh ../data/dev.json outputs/infer
```

This will automatically execute `gpt_ambiguous.py`, `gpt_neither_fits.py`, `gpt_preference.py`, `gpt_correct.py`, and `gpt_rating.py` in sequence, saving all results to the `outputs/` directory.

### Regression & Prediction

Once you have generated the GPT features for both your train and inference sets, you can train the XGBoost regression model and generate final predictions in a single step using our bash script.

To run the regression pipeline, execute the script by passing your train data, inference data, and desired output filename:

```bash
# Make the script executable (only needed once)
chmod +x run_regression.sh

# Run the training, prediction, and formatting pipeline
./run_regression.sh ../data/train.json ../data/dev.json final_task_decomp_predictions.jsonl
```

This script will automatically:

1. Train the regression model (`xgb_regression.pkl`) using the features in `outputs/train/`.
2. Run inference on the dev set using the features in `outputs/infer/`, outputting a temporary CSV.
3. Convert the CSV into the official `.jsonl` submission format.
