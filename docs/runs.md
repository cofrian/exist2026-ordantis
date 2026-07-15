# The 18 submitted runs → scripts & configuration

Ordantis submitted 3 runs × 2 settings (hard/soft) × 3 subtasks = **18 files**, all in
[`../submissions/`](../submissions/). The table below maps each official run to the model that
produced it and its post-processing.

**Naming key** (script suffixes in `src/`):
- `max512` — XLM-RoBERTa-base, max length 512
- `longformer` — multilingual Longformer (`markussagen/xlm-roberta-longformer-base-4096`), max length ~1100
- `_R` — enriched text **includes** the Gemini `reasoning` field
- `_v2` — alternative seed/config variant · `_full` — final retrain on **train + val** combined
- `blend` — linear combination of the trained model's probability with Gemini's zero-shot probability

> The model↔run association below is Ordantis' own reconstruction (each row was matched against the
> exact submitted file; match % was ≥ ~93% except two 2.2 soft Platt variants). Blends and thresholds
> are applied on top of the listed base checkpoint.

## Task 2.1 — Binary sexism detection

| File | Base model (script) | Post-processing |
|---|---|---|
| `task2_1_hard_Ordantis_1` | `task21_max512_R.py` (XLM-R, +reasoning) | threshold 0.46 |
| `task2_1_hard_Ordantis_2` | `task21_longformer.py` | threshold 0.39 |
| `task2_1_hard_Ordantis_3` | `task21_max512.py` | threshold 0.51 |
| `task2_1_soft_Ordantis_1` 🥇 | GEMF (`M3_vista_E` base) | **blend** `0.6·P_model + 0.4·P_Gemini` on P(YES) |
| `task2_1_soft_Ordantis_2` | `task21_longformer.py` | direct probabilities |
| `task2_1_soft_Ordantis_3` | `task21_longformer_R.py` (+reasoning) | direct probabilities |

## Task 2.2 — Source-intention classification (hierarchical head)

| File | Base model (script) | Post-processing |
|---|---|---|
| `task2_2_hard_Ordantis_1` | `task22_longformer.py` | thresholds (0.425, 0.35) |
| `task2_2_hard_Ordantis_2` | `task22_max512.py` | thresholds (0.475, 0.300) |
| `task2_2_hard_Ordantis_3` 🥇 | `task22.py` base (`task22_best`) | **blend** + thresholds (0.475, 0.500) |
| `task2_2_soft_Ordantis_1` 🥇 | `task22.py` base (`task22_best`) | **raw blend** `0.6·model + 0.4·Gemini` |
| `task2_2_soft_Ordantis_2` | `task22_longformer.py` | per-class Platt scaling |
| `task2_2_soft_Ordantis_3` | `task22_max512.py` | per-class Platt scaling |

## Task 2.3 — Multi-label categorization (conditional head)

| File | Base model (script) | Post-processing |
|---|---|---|
| `task2_3_hard_Ordantis_1` | `task23_max512_v2.py` | category threshold 0.15 (argmax if empty) |
| `task2_3_hard_Ordantis_2` | `task23_max512.py` | category threshold 0.15 (argmax if empty) |
| `task2_3_hard_Ordantis_3` | `task23_max512_R.py` (+reasoning) | category threshold 0.15 (argmax if empty) |
| `task2_3_soft_Ordantis_1` | `task23_longformer.py` (no sampler) | direct probabilities |
| `task2_3_soft_Ordantis_2` | `task23_max512.py` | direct probabilities |
| `task2_3_soft_Ordantis_3` | `task23_longformer_v2.py` (sampler) | direct probabilities |

🥇 = ranked **first** in its official leaderboard setting.

## Final retrain scripts

`task21_full.py`, `task22_longformer_full.py` and `task23_longformer_full.py` retrain the chosen
architectures on **train + val combined** (fixed epoch budget, no early stopping) to produce the
final checkpoints used for the test predictions.
