# SemEval 2026 Task 5


## GPT Subtask and Rating Model

First go to the `SemEval2026-WSP/korean/gpt_subtask_and_rating` folder and install the requirements in `requirements.txt`

`pip install -r requirements.txt`

We first collect subtask features by prompting GPT-5 on the train and dev sets. Run these scripts on both `train.json` and `dev.json`.

### Prompting

`gpt_ambiguous.py`

To run first go to the `SemEval2026-WSP/korean/gpt_subtask_and_rating` folder then run ``python prompts/gpt_ambiguous.py --in <YOUR DATA FILE> --out_json outputs``


`gpt_neither_fits.py`

Command to run: ``python prompts/gpt_neither_fits.py --in <YOUR DATA FILE> --out_json outputs``


`gpt_preference.py`

Command to run:  ``python prompts/gpt_preference.py --in <YOUR DATA FILE> --out_json outputs``


`gpt_correct.py`

Command to run:  ``python prompts/gpt_correct.py --in <YOUR DATA FILE> --out_json outputs``


`gpt_rating.py`

Command to run:  ``python prompts/gpt_rating.py --in <YOUR DATA FILE> --out_json outputs``


### Classifier

To train the classifier go to the `SemEval2026-WSP/korean/gpt_subtask_and_rating` folder. 


`train_classification.py`

Command to run: `python classifier/train_classification.py --train_json <YOUR DATA FILE> --gpt_preference_json outputs/train/train_preference.json --ambig_json outputs/train/train_ambiguity.json --neither_json outputs/train/train_neither.json --correct_json outputs/train/train_correct.json --rating_json outputs/train/train_rating.json`


You will also get the fitted model `xgb.pkl` as output. Then to predict using the classifier:

`predict_classification`

Command to run: `python classifier/predict_classification.py --dev_json <YOUR DATA FILE> --gpt_preference_json outputs/dev/dev_preference.json --ambig_json outputs/dev/dev_ambiguity.json --neither_json outputs/dev/dev_neither.json --correct_json outputs/dev/dev_correct.json --rating_json outputs/dev/dev_rating.json --model xgb.pkl --out dev_predictions.csv`


You will get `dev_predictions.csv` as output. You can convert it to the prediction format with the command:

`python tools/csv_to_jsonl.py --in_csv dev_predictions.csv --out_jsonl <YOUR FILE NAME.jsonl>`


### Regression

To train the regression model go to the `SemEval2026-WSP/korean/gpt_subtask_and_rating` folder.


`train_regression.py`

Command to run: `python classifier/train_regression.py --train_json <YOUR DATA FILE> --gpt_preference_json outputs/train/train_preference.json --ambig_json outputs/train/train_ambiguity.json --neither_json outputs/train/train_neither.json --correct_json outputs/train/train_correct.json --rating_json outputs/train/train_rating.json`


You will also get the fitted model `xgb_regression.pkl` as output. Then to predict using the regression model:

`predict_regression.py`

Command to run: `python classifier/predict_regression.py --dev_json <YOUR DATA FILE> --gpt_preference_json outputs/dev/dev_preference.json --ambig_json outputs/dev/dev_ambiguity.json --neither_json outputs/dev/dev_neither.json --correct_json outputs/dev/dev_correct.json --rating_json outputs/dev/dev_rating.json --model xgb_regression.pkl --out dev_predictions_regression.csv`


You will get `dev_predictions_regression.csv` as output. You can convert it to the prediction format with the command:

`python tools/csv_to_jsonl.py --in_csv dev_predictions_regression.csv --out_jsonl <YOUR FILE NAME.jsonl>`


## Qwen Ambiguity Rating

To run go to `SemEval2026-WSP/korean/qwen_ambiguity_rating`


### Prompting

`qwen_oracle.py`

Then to prompt qwen run the following: `python qwen_oracle.py --data_json <YOUR DATA FILE> --out_dir outputs_oracle`


### Outputs

`outputs_oracle/<key>.json` contains the individual qwen judgements whether the sample has a clear or ambiguous meaning.

### Converting to predictions

Run: `python oracle_to_predictions.py --oracle_dir outputs_oracle --data <YOUR DATA FILE> --out <PREDICTION FILE.jsonl>`
