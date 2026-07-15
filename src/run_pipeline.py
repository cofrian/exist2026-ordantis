"""Pipeline principal EXIST 2026 — subtarea 2.1 (memes).

Estado actual: Fase 1 — datos + preprocesado + embeddings ViT + M1 entrenable
con threshold óptimo (Mejora B) y temperature scaling (Mejora C) en validación.
M2, M3 y el resto de mejoras (D, E, F) se añaden a continuación una vez M1 da número.
"""
import json
import os

import numpy as np

import config as C
import data as D
import evaluation_utils as E
from precompute import precompute_vit_embeddings


def main():
    print("=" * 60)
    print(f"EXIST 2026 — Task 2.1  |  DRY_RUN={C.DRY_RUN}  |  TEAM={C.TEAM_NAME}")
    print("=" * 60)

    # 1) Setup
    C.set_seed(C.SEED)
    C.configure_gpu()

    # 2) Carga + preprocesado + split
    print("\n[1] Cargando y preprocesando datos ...")
    splits = D.load_split()
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    # cache de stats y textos limpios
    json.dump(splits["sensor_stats"], open(os.path.join(C.PRE_DIR, "sensor_stats.json"), "w"))
    json.dump({e["id"]: e["text"] for e in splits["train"] + splits["val"] + splits["test"]},
              open(os.path.join(C.PRE_DIR, "text_clean.json"), "w"), ensure_ascii=False)

    # 3) Embeddings ViT (todos los memes)
    print("\n[2] Pre-calculando embeddings ViT ...")
    all_ex = splits["train"] + splits["val"] + splits["test"]
    vit_emb = precompute_vit_embeddings(all_ex)

    # 4) Entrenar M1
    from train import train_model
    bt, bi = C.BATCH["M1"]
    if C.DRY_RUN:
        bt, bi = 8, 16
    cfg_m1 = dict(text=True, image=True, et=True, hr=True, eeg=True, caption=False, set_pool=False)
    res = train_model("M1", cfg_m1, bt, bi, seed=C.SEED, splits=splits, vit_emb=vit_emb)

    # 5) Baselines + métricas de validación
    print("\n[3] Baselines oficiales en validación ...")
    bl = E.baseline_metrics(res["val_ids"], res["val_targets"])
    for k, v in bl.items():
        print(f"  {k}: ICM={v.get('ICM')}  FMeasure={v.get('FMeasure')}")

    val_probs = 1.0 / (1.0 + np.exp(-res["val_logits"]))
    m1_hard = E.eval_hard(res["val_ids"], val_probs >= 0.5, res["val_targets"])
    m1_soft = E.eval_soft(res["val_ids"], val_probs, res["val_targets"])
    print(f"\n[M1] val (thr 0.5): ICM={m1_hard.get('ICM')}  FMeasure={m1_hard.get('FMeasure')}  "
          f"ICMSoft={m1_soft.get('ICMSoft')}")
    if m1_hard.get("ICM") is not None and m1_hard["ICM"] <= max(
            bl["baseline_majority"].get("ICM", -1e9), bl["baseline_minority"].get("ICM", -1e9)):
        print("  *** WARNING: M1 NO supera los baselines oficiales en ICM ***")

    # 6) Mejora B — threshold óptimo
    print("\n[4] Optimizando threshold con ICM (Mejora B) ...")
    best_t, best_icm, hist = E.optimize_threshold(res["val_ids"], val_probs, res["val_targets"])
    print(f"  threshold óptimo M1 = {best_t}  (ICM={best_icm:.4f})")
    json.dump({"M1": best_t}, open(os.path.join(C.VAL_DIR, "thresholds_optimos.json"), "w"))

    # 7) Mejora C — temperature scaling
    print("\n[5] Temperature scaling (Mejora C) ...")
    T = E.fit_temperature(res["val_logits"], res["val_targets"])
    print(f"  temperature M1 = {T:.4f}")
    cal_probs = E.apply_temperature(res["val_logits"], T)
    m1_soft_cal = E.eval_soft(res["val_ids"], cal_probs, res["val_targets"])
    print(f"  ICMSoft tras calibración: {m1_soft_cal.get('ICMSoft')} "
          f"(antes {m1_soft.get('ICMSoft')})")
    json.dump({"M1": T}, open(os.path.join(C.VAL_DIR, "temperature_scaling.json"), "w"))

    # guardar predicciones de validación
    D.write_hard_file(os.path.join(C.VAL_DIR, "M1_val_predictions_hard.json"),
                      res["val_ids"], val_probs >= best_t)
    D.write_soft_file(os.path.join(C.VAL_DIR, "M1_val_predictions_soft.json"),
                      res["val_ids"], cal_probs)

    print("\n" + "=" * 60)
    print("FASE 1 COMPLETADA — M1 entrenado y evaluado en validación")
    print(f"  M1: ICMSoft={res['best_metrics'].get('val_icm_soft')}  "
          f"ICM={res['best_metrics'].get('val_icm')}  FMeasure={res['best_metrics'].get('val_f1')}  "
          f"AUC={res['best_metrics'].get('val_auc')}")
    print(f"  threshold óptimo={best_t}  temperature={T:.3f}")
    print("=" * 60)
    return res


if __name__ == "__main__":
    main()
