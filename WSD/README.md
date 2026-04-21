# ConSec WSD Integration

This utility provides an interactive interface for performing Word Sense Disambiguation (WSD) using the **ConSec** framework.
The following steps are tested with python 3.8.20.

### 1. Environment Setup
First, install the necessary Python dependencies:

```bash
python -m pip install -r requirements.txt
```

You may also need to install the following packages separately to convince pip to allow a slight mismatch of requirements; ConSec has some deprecations today.

```bash
python -m pip install tokenizers
python -m pip pip install torch==1.7.1 torchtext==0.8.1 pytorch-lightning==1.1.7
```

### 2. Install ConSec
This tool requires the official ConSec implementation. Clone it and follow their setup instructions:
* **Repository:** [sapienzanlp/consec](https://github.com/sapienzanlp/consec)

---

## Usage

Run `queryconsec.py` by passing the required file paths and configuration. 

### Arguments
| Argument | Description |
| :--- | :--- | 
| `--input_file` | Input file in the JSON format provided by task organizers. |
| `--output_file` | The path where the resulting `.jsonl` file will be saved. |
| `--consec_location` | The absolute path to your local ConSec root directory. |
| `--consec_predict_file` | Relative path to ConSec's `predict.py` from the root. |
| `--ckpt` | Relative path to the checkpoint (`.ckpt`) file from the root. |

> The default relative paths for the prediction script and checkpoint should work as-is if you followed the standard ConSec installation instructions

---

## Example

```bash 
python3 queryconsec.py \
  --consec_location /home/user/consec/ \
  --input_file dev.json \
  --output_file wsd_predictions.jsonl
```


