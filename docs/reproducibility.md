# Reproducibility guide

This guide reproduces the Ordantis EXIST 2026 Task 2 pipeline end to end.

## 0. Hardware & environment

- The paper's runs used a single **NVIDIA RTX 4500 Ada (25.3 GB)** in **BF16**. Any ≥16 GB CUDA GPU
  should work; CPU-only is possible for the smoke test but not for full training.
- If `flash-attn` is not installed the code automatically falls back to PyTorch `sdpa` attention.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get the data

The **EXIST 2026 dataset is not redistributable**. Download it from the
[organizers](http://nlp.uned.es/exist2026/) and lay it out so `src/config.py` finds it:

```
exist2026-ordantis/
├── datos/          # git-ignored; any nesting is fine, config.py globs for it
│   └── **/<... Memes Dataset>/training/EXIST2026_training.json
│                              /test/EXIST2026_test_clean.json
│   └── **/evaluation/exist2025_format_val_V0.2.py   # official format validator
└── src/
```

`config.py` auto-detects folders whose names contain `Memes`/`Dataset` and is robust to spaces vs.
underscores. Generated outputs go to `src/exist2026_Ordantis/` (checkpoints, preprocessed caches,
validation results) — all git-ignored.

## 2. Configure the Gemini key (offline precomputation only)

```bash
cp .env.example .env      # then set GEMINI_API_KEY=...
```

Without a key the Gemini scripts exit gracefully (`sys.exit(0)`) and the pipeline runs on the
non-Gemini features only — but the headline results depend on the Gemini enrichment, so a key is
needed to reproduce them.

## 3. Offline precomputation (cached, one-off)

```bash
cd src
python precompute_emotions.py     # Ekman 7-dim emotion features from OCR  → preprocessed/ekman_emotions.json
python precompute_gemini.py       # one Gemini call per meme (3 subtasks)   → preprocessed/gemini_predictions.json
# optional, Task 2.3 augmentation:
python generate_paraphrases_task23.py
# baseline visual branch (only if you want to reproduce the M1/M2 ViT baselines):
python precompute.py
```

The Gemini step is the dominant one-off cost (~74 min / ~$47 for 5037 memes in the paper). It is
**incrementally cached**: re-running resumes from `gemini_predictions.json` and skips finished memes.

## 4. Smoke test (minutes)

```bash
DRY_RUN=1 python run_full.py      # tiny subset, validates the whole pipeline gives numbers
```

## 5. Reproduce a specific run

Each of the 18 submissions maps to a script in [`runs.md`](runs.md). Examples:

```bash
# Task 2.1 hard run 1 (XLM-R + reasoning):
python task21_max512_R.py

# Task 2.2 hard/soft (Longformer, hierarchical head):
python task22_longformer.py

# Task 2.3 soft run 1 (Longformer, no sampler — best soft):
python task23_longformer.py

# Final retrains on train+val for the submitted checkpoints:
python task21_full.py
python task22_longformer_full.py
python task23_longformer_full.py
```

Useful environment flags (see `config.py`): `DRY_RUN=1`, `FORCE_RECOMPUTE=1`, `USE_COMPILE=1`
(torch.compile, off by default), `STRICT_PHASE2=0/1`.

## 6. Evaluation

Metrics use the **official PyEvALL** framework (ICM / ICM-Soft) via `evaluation_utils.py`. Note the
paper's distinction: development metrics (F1/AUC, local threshold search) guided design, but the
final ranking criterion is the PyEvALL-based official score. The already-submitted files live in
[`../submissions/`](../submissions/) for comparison.

## 7. Post-hoc analysis (paper tables & figures)

The `src/analysis/` scripts reproduce the calibration (ECE/MCE/Brier), per-category error analysis,
co-occurrence and modality-ablation tables. They add the source directory to `sys.path`
automatically, so run them from inside their folder, e.g.:

```bash
cd src/analysis/results && python bloque1_calibracion.py
cd src/analysis/review  && python tarea4_ablacion.py
```

## Notes & caveats

- Determinism: seeds are set (`config.SEED=42`) but `cudnn.benchmark=True` and BF16 mean results are
  reproducible up to small numerical noise, not bit-exact.
- Calibration numbers (Platt/temperature) are fit and evaluated on the same validation split, so
  they are optimistic upper bounds (paper §13).
- Task 2.3 hard: the category thresholds overfit validation and do not transfer to test; the soft
  ranking (10/118) is the fair reflection of the model quality (paper §14).
