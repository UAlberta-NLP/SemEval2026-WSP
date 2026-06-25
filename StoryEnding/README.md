# Story Ending

### 1. Environment Setup
First, install the necessary Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run `story.py` by passing the required file paths and configuration. 

### Arguments
| Argument | Description |
| :--- | :--- | 
| `--train_path` | Input file address in the JSON format provided by task organizers. |
| `--infer_path` | File to perform inference on, also in the JSON format provided by the organizers. |
| `--output_location` | The path to output our inferences to. |

> The default relative paths should work as-is.

---

## Example

```bash 
python3 story.py \
  --train_path ../data/train.json \
  --infer_path ../data/test.json \
  --output_location ../predictions
```


