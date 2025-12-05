# SemEval2026-WSP
Official repository for UAlberta at SemEval-2026 Task 5: Rating Word Senses Plausibility (WSP) in Ambiguous Sentences through Narrative Understanding.

🔗 [Task Page](https://nlu-lab.github.io/semeval.html) | 💻 [Official Scripts](https://github.com/Janosch-Gehring/semeval26-05-scripts)

---

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Run Baselines
```bash
bash baselines.sh
```

---

## 📁 Directory Overview

- **`src/`** — Source code for baselines and evaluation scripts
- **`res/`** — Resources including data, predictions, and scores
- **`baselines.sh`** — Script to run all baselines and generate results
- **Team Folders** — Language-specific directories for team submissions:
  - **`chinese/`** — Chinese team workspace
  - **`korean/`** — Korean team workspace
  - **`persian/`** — Persian team workspace
  - **`urdu/`** — Urdu team workspace

---

## 👥 Team Workflow

### For Students

**IMPORTANT:** Each team must work exclusively in their designated language folder to avoid merge conflicts.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/UAlberta-NLP/SemEval2026-WSP.git
   cd SemEval2026-WSP
   ```

2. **Work in your team folder:**
   - **Chinese team** → work only in `chinese/`
   - **Korean team** → work only in `korean/`
   - **Persian team** → work only in `persian/`
   - **Urdu team** → work only in `urdu/`

3. **Create your model files:**
   ```bash
   cd <your-team-folder>
   # Add your code, notebooks, models, etc.
   ```

4. **Commit and push your changes:**
   ```bash
   git add <your-team-folder>/
   git commit -m "Add <description> for <language> team"
   git push origin main
   ```

5. **Pull before pushing:**
   ```bash
   git pull origin main
   git push origin main
   ```

### Accessing Shared Resources

All teams can access shared resources in the `res/` and `src/` directories:
- Training data: `res/data/train.json`
- Dev data: `res/data/dev.json`
- Evaluation scripts: `src/eval/`
- Baseline implementations: `src/baselines/`

**Note:** Do not modify files outside your team folder unless coordinated with the instructor.

---

## 👥 Author

- **Ning Shi** — <mrshininnnnn@gmail.com>
 