# UAlberta at SemEval-2026 Task 5: Disambiguating Stories via Task Decomposition

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square)](assets/paper.pdf)
[![Poster](https://img.shields.io/badge/Poster-PDF-blue?style=flat-square)](assets/poster.pdf)
[![Slides](https://img.shields.io/badge/Slides-PDF-green?style=flat-square)](assets/slides.pdf)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?style=flat-square)](https://youtu.be/ip5St-5ulsI)
[![Task](https://img.shields.io/badge/Task-Word%20Sense%20Plausibility-orange?style=flat-square)](https://nlu-lab.github.io/semeval.html)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-CodaBench-yellow?style=flat-square)](https://www.codabench.org/competitions/10877/#/results-tab)

This repository contains our system for **SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Stories through Narrative Understanding**. Given a short narrative containing an ambiguous word (a *homonym*) with two candidate senses, the task is to predict, on a 1–5 scale, how plausible each sense is in context. Our approach centers on **task decomposition (TD)**: rather than predicting a score directly, we break the problem into simpler subtasks and combine their outputs, then ensemble complementary signals from word sense disambiguation, fine-tuned embeddings, and large language models.

:second_place_medal: **2nd Place on the Official Leaderboard** | **0.840 Spearman Correlation** (comparable to the 0.834 estimated human upper bound)

---

## Overview

The input story is processed in parallel by several components — task decomposition, direct LLM prompting, word sense disambiguation, a fine-tuned story-ending model, and a translation-based method (one homonym per translation). Their outputs are combined with a ridge regression ensemble to produce the final plausibility score. Task decomposition is the strongest individual component, accounting for more than half of the ensemble's contribution.

---

## Repository Structure

```
.
├── TaskDecomposition/   # TD: binary-decision prompts + regressor
├── DirectPrompting/     # Direct LLM prompting variants (e.g., Qwen)
├── WSD/                 # ConSec continuous WSD interface
├── StoryEnding/         # Fine-tuned DeBERTa story-ending model
├── OHPT/                # One Homonym Per Translation
├── Ensemble/            # Ridge regression ensemble + scaling
└── assets/              # Paper, poster, and slides
```

Each component directory contains its own README with setup and usage instructions.

---

## Citation

```bibtex
@inproceedings{basil-etal-2026-ualberta,
    title = "{UA}lberta at {S}em{E}val-2026 Task 5: Disambiguating Stories via Task Decomposition",
    author = "Basil, David  and
      Cho, Junhyeon  and
      Girigowda, Chirooth  and
      Luo, Guoqing  and
      Momin, Sahir  and
      Robinson, Sevryn  and
      Shi, Ning  and
      Kondrak, Grzegorz",
    booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)",
    year = "2026",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
}
```
