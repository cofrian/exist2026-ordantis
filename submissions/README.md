# Submitted runs

The 18 official Ordantis files submitted to EXIST 2026 Task 2 (PyEvALL format).

- Naming: `task2_<subtask>_<setting>_Ordantis_<run>` (subtask 1/2/3, setting `hard`/`soft`, run 1–3).
- **Hard** files: one label per meme, e.g. `{"test_case": "EXIST2025", "id": "310001", "value": "YES"}`.
- **Soft** files: a probability distribution, e.g. `... "value": {"YES": 0.637, "NO": 0.363}`.

Each file maps to a model and post-processing config in [`../docs/runs.md`](../docs/runs.md).
Runs that ranked **first** in their leaderboard setting: `task2_1_soft_Ordantis_1`,
`task2_2_hard_Ordantis_3`, `task2_2_soft_Ordantis_1`.
