# Post-hoc analysis

Scripts and result tables behind the paper's analysis sections. These are **not** part of the
prediction pipeline; they consume cached predictions and produce the tables/figures.

- **`results/`** — calibration (ECE/MCE/Brier), Task 2.3 per-category & co-occurrence error
  analysis, Gemini-feature ablation, parameter counts. The `bloque*.py` scripts are the main entry
  points (`bloque1_calibracion.py`, `bloque2_errores23.py`, `bloque3_ablacion_gemini.py`, …);
  `_*.py` files are helper/recompute utilities. Result CSVs and `figuras/` are checked in.
- **`review/`** — cross-run metric consolidation and per-variant evaluation
  (`consolidate_metrics.py`, `eval_*_variants.py`, `metricas_por_run.py`, `tarea*_*.py`), plus the
  HTML/LaTeX review report and `figuras/`.

Each script inserts the source root into `sys.path`, so run them from inside their own folder. They
require the cached predictions / checkpoints produced by the pipeline (git-ignored), so they are
provided for transparency of the paper's numbers rather than as a one-command reproduction.
