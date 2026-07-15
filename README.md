# GEMF — Gemini-Enriched Multimodal Fusion for Sexism in Memes

**Ordantis submission to [EXIST 2026](http://nlp.uned.es/exist2026/) — Task 2 (Sexism Characterization in Memes).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-CLEF%202026-b31b1b.svg)](paper/paper_562.pdf)

This repository contains the full pipeline behind the Ordantis runs for EXIST 2026 Task 2:
binary sexism detection (2.1), source-intention classification (2.2) and multi-label sexism
categorization (2.3), over **bilingual (EN/ES) memes enriched with physiological data**.

The core idea is to use **Gemini as a semantic mediator** rather than a black-box classifier:
Gemini interprets each meme offline and turns it into structured text (description, sexism
analysis, reasoning, intention/irony cues and zero-shot probabilities). That text is concatenated
with the OCR, encoded with **XLM-RoBERTa** or a **multilingual Longformer**, and fused with **EEG**
and **Ekman emotion** features before a probabilistic decision. Training uses **soft (annotator-
distribution) targets** rather than majority labels (Learning with Disagreement).

> 📄 Full write-up: [`paper/paper_562.pdf`](paper/paper_562.pdf) ·
> Detailed technical report (Spanish): [`docs/informe_es.md`](docs/informe_es.md)

---

## Official results (all-instances)

| Subtask | Setting | Best run | Rank | Metric |
|---|---|---|---|---|
| 2.1 Binary | Soft | `Ordantis_1` (Gemini blend) | **1 / 144** | ICM-Soft 0.7206 |
| 2.1 Binary | Hard | `Ordantis_2` (Longformer)   | 3 / 217 | ICM-Hard 0.4079, F1-YES 0.801 |
| 2.2 Intention | Hard | `Ordantis_3` (model–Gemini blend) | **1 / 186** | ICM-Hard 0.3709, F1 0.616 |
| 2.2 Intention | Soft | `Ordantis_1` (raw blend) | **1 / 117** | ICM-Soft 0.0114 |
| 2.3 Categories | Soft | `Ordantis_1` (Longformer, no sampler) | 10 / 118 | ICM-Soft Norm 0.2516 |
| 2.3 Categories | Hard | `Ordantis_1` | 132 / 187 | F1 0.379 |

The Gemini-based semantic enrichment is the single largest design gain (Task 2.1 validation AUC
**0.741 → 0.880**). Task 2.3 hard remained challenging: the underlying probabilistic model ranks
well (soft rank 10/118) but the category-threshold binarization does not transfer to the test
distribution — a post-processing rather than a modeling problem (see the paper, §14).

---

## Architecture (GEMF)

```
                         ┌─────────────────────────────────────────────┐
   Meme image  ─────────▶│  Gemini (OFFLINE, one call per meme)         │
                         │  → description, sexism analysis, reasoning,  │
                         │    intention/irony cues, zero-shot probs     │
                         └───────────────┬─────────────────────────────┘
                                         │ enriched text        │ numeric features / probs
   OCR text ──────┐                      ▼                      │
                  └──▶ concat ──▶ XLM-RoBERTa (max 512)          │ (auxiliary features
                                  or mult. Longformer (max 1100) │  and optional blend)
                                  + masked mean pooling          │
                                  → h_text ∈ ℝ⁷⁶⁸                │
   EEG {subjects} ──▶ Set Attention Pooling  → h_EEG ∈ ℝ²⁵⁶      │
   Ekman emotions ──▶ 7-dim probs            → h_emo ∈ ℝ⁷        │
                                  │                              │
                                  ▼  concat (+ Gemini features)  │
                          Shared MLP trunk ◀─────────────────────┘
                                  │
                 ┌────────────────┼────────────────────┐
        2.1 Binary head   2.2 Hierarchical head   2.3 Conditional multi-label head
        P(YES)=σ(·)       P(NO)=1−pₛ; DIRECT/JUDG  P(cᵢ)=P(sexist)·P(cᵢ|sexist)
```

See [`docs/architecture.md`](docs/architecture.md) for the full description and the fusion
dimensions (ℝ¹⁰³¹ / ℝ¹⁰³⁸ / ℝ¹⁰³⁷).

---

## Repository structure

```
exist2026-ordantis/
├── README.md
├── requirements.txt        # Python dependencies
├── .env.example            # template for the Gemini API key (copy to .env)
├── paper/                  # CLEF 2026 working-notes paper (paper_562.pdf)
├── submissions/            # the 18 official Ordantis run files (the deliverable)
├── docs/
│   ├── architecture.md     # GEMF architecture in detail
│   ├── reproducibility.md  # step-by-step reproduction guide
│   ├── runs.md             # mapping: each of the 18 runs → script + config
│   ├── gemini_prompt.md    # exact offline Gemini prompt (paper Appendix A)
│   └── informe_es.md       # detailed engineering report (Spanish)
└── src/
    ├── config.py           # paths (auto-detected), hyper-params, seeds, GPU setup
    ├── data.py             # JSON loading, OCR cleaning, soft targets, sensor z-scoring, splits
    ├── dataset.py          # MemeDataset + collate (subject padding with mask)
    ├── models.py           # MemeClassifier, SetAttentionPool, masked mean pooling
    ├── train.py            # two-phase fine-tuning (frozen warm-up → joint), early stopping
    ├── inference.py        # test-time inference / TTA
    ├── evaluation_utils.py # PyEvALL wrappers (ICM / ICM-Soft), thresholds, calibration
    ├── precompute.py             # ViT embeddings (baseline visual branch)
    ├── precompute_gemini.py      # OFFLINE Gemini enrichment (one call/meme, 3 subtasks)
    ├── precompute_emotions.py    # Ekman 7-dim emotion features from OCR
    ├── generate_paraphrases_task23.py  # optional Task 2.3 data augmentation
    ├── run_full.py / run_pipeline.py / run_all.sh   # orchestration
    ├── task21_*.py / task22_*.py / task23_*.py      # the run scripts (see docs/runs.md)
    └── analysis/           # post-hoc analysis behind the paper tables
        ├── results/        # calibration, error analysis, ablation, per-category metrics
        └── review/         # metric consolidation, per-run evaluation, figures
```

> **Naming key** for the run scripts: `max512` = XLM-RoBERTa-base @ 512 tokens · `longformer`
> = multilingual Longformer @ 1100 tokens · `_R` = includes Gemini `reasoning` field ·
> `_full` = final retrain on train+val. Full mapping in [`docs/runs.md`](docs/runs.md).

---

## Installation

```bash
git clone https://github.com/cofrian/exist2026-ordantis.git
cd exist2026-ordantis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended (the paper's runs used a single NVIDIA RTX 4500 Ada, 25 GB, BF16).

## Data

The **EXIST 2026 dataset is not redistributable** and is not included here. Download it from the
[organizers](http://nlp.uned.es/exist2026/) and place it so that `src/config.py` can auto-detect it:

```
exist2026-ordantis/
├── datos/                              # <-- put the dataset here (git-ignored)
│   └── .../<Memes Dataset>/training/EXIST2026_training.json
│                          /test/EXIST2026_test_clean.json
└── src/
```

`config.py` auto-resolves paths robustly to spaces/underscores in folder names.

## API key (only for Gemini precomputation)

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...   (never commit .env — it is git-ignored)
```

The Gemini step is a **one-off offline cost** (~74 min / ~$47 for the full 5037-meme corpus in the
paper) and is cached, so every downstream run reuses it at no extra cost. If no key is present the
Gemini scripts exit gracefully and the pipeline falls back to non-Gemini features.

---

## Reproduce

See [`docs/reproducibility.md`](docs/reproducibility.md) for the full guide. In short:

```bash
cd src

# 1) Offline precomputation (cached)
python precompute_emotions.py          # Ekman emotion features
python precompute_gemini.py            # Gemini enrichment (requires ../.env)

# 2) Quick end-to-end smoke test (minutes)
DRY_RUN=1 python run_full.py

# 3) A specific run — e.g. the Task 2.1 soft blend that ranked #1
python task21_max512_R.py              # (see docs/runs.md for the run↔script table)
```

Evaluation uses **PyEvALL** (official ICM / ICM-Soft). The 18 files already submitted are in
[`submissions/`](submissions/) for reference; the scripts regenerate them under
`src/exist2026_Ordantis/` (git-ignored).

---

## Citation

```bibtex
@inproceedings{ortiz2026gemf,
  title     = {Gemini-Enriched Probabilistic Modeling for Multimodal Sexism Characterization in Memes},
  author    = {Ortiz Montesinos, Sergio and Mart{\'i}nez G{\'o}mez, Fernando},
  booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the Evaluation Forum},
  year      = {2026}
}
```

## Acknowledgements & responsible-AI note

Developed by the **Ordantis** team for EXIST 2026. This project studies the *detection* of sexism;
all model outputs (including Gemini's descriptions and reasoning) are auxiliary signals, not ground
truth, and should not be treated as verified statements. See the paper's Limitations (§17) for the
single-interpreter dependence and calibration caveats.

Generative-AI assistants (Claude, Anthropic) were used for grammar, translation and code debugging
from author-provided specifications; the authors reviewed all content and take full responsibility.

## License

Code: [MIT](LICENSE). Dataset: EXIST 2026 organizers' terms. Paper: CC BY 4.0.
