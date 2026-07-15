"""Pipeline COMPLETO EXIST 2026 — subtarea 2.1 (memes): M1 + M2 + M3 (5 vistas),
Mejoras A-F, inferencia con TTA, calibración, thresholds óptimos, 6 submissions,
validación de formato y empaquetado del zip.

Uso:
    python run_full.py            # corrida real
    DRY_RUN=1 python run_full.py  # prueba rápida end-to-end
"""
import csv
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

import config as C
import data as D
import evaluation_utils as E
from precompute import precompute_vit_embeddings


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _wait_for_gemini(timeout_min=90):
    """Espera al precompute de Gemini que corre en background (run_all.sh).

    Termina cuando: aparece el marcador .DONE, o el proceso ha muerto sin marcador
    (sin API key / crash), o se agota el timeout. Si no hay pid ni marcador (p.ej.
    se ejecuta run_full.py suelto), no espera nada.
    """
    done_marker = os.path.join(C.PRE_DIR, "gemini_predictions.DONE")
    pid_file = os.path.join(C.PRE_DIR, "gemini.pid")
    if os.path.exists(done_marker):
        return
    pid = None
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file).read().strip())
        except Exception:
            pid = None
    if pid is None and not os.path.exists(done_marker):
        return  # nadie nos dijo que esperásemos
    print(f"[Gemini] esperando a que termine el precompute (pid={pid}, timeout={timeout_min}min) ...")
    cache = os.path.join(C.PRE_DIR, "gemini_predictions.json")
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        if os.path.exists(done_marker):
            print(f"[Gemini] precompute terminado ({open(done_marker).read().strip()} válidas).")
            return
        if pid is not None and not _pid_alive(pid):
            print("[Gemini] el proceso ha terminado sin marcador .DONE -> sigo con lo que haya.")
            return
        n = 0
        try:
            n = len(json.load(open(cache))) if os.path.exists(cache) else 0
        except Exception:
            pass
        print(f"[Gemini] ... {n} memes procesados ({(time.time()-t0)/60:.0f} min esperando)", flush=True)
        time.sleep(20)
    print("[Gemini] timeout esperando -> sigo con lo que haya.")


def load_gemini_captions():
    """Construye {id -> texto descriptivo} a partir de gemini_predictions.json.

    Usamos la 'description' (descripción literal de imagen+texto) y el 'sexism_analysis'
    que devuelve Gemini 3.1 Pro como "caption" para la Vista E de M3 — en lugar de los
    captions de Qwen-VL. Devuelve None si el fichero no existe o no hay entradas válidas.
    """
    path = os.path.join(C.PRE_DIR, "gemini_predictions.json")
    if not os.path.exists(path):
        return None
    try:
        preds = json.load(open(path, encoding="utf-8"))
    except Exception as ex:
        print(f"  [aviso] no se pudo leer {path}: {ex}")
        return None
    caps = {}
    for mid, v in preds.items():
        if not isinstance(v, dict):
            continue
        desc = (v.get("description") or "").strip()
        analysis = (v.get("sexism_analysis") or "").strip()
        parts = []
        if desc:
            parts.append(f"Description: {desc}")
        if analysis:
            parts.append(f"Sexism Analysis: {analysis}")
        if parts:
            caps[str(mid)] = " ".join(parts)
    return caps or None


def _auc(soft_targets, probs):
    try:
        from sklearn.metrics import roc_auc_score
        y = (np.asarray(soft_targets) >= 0.5).astype(int)
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, probs))
    except Exception:
        return None


def _row(ids, soft_targets, hard_probs, hard_thr, soft_probs, label):
    """Una fila de la tabla de ablación: AUC + F1+ + métricas hard/soft de PyEvALL."""
    h = E.eval_hard(ids, np.asarray(hard_probs) >= hard_thr, soft_targets)
    s = E.eval_soft(ids, soft_probs, soft_targets)
    return dict(Mejoras=label, AUC=_auc(soft_targets, soft_probs),
                F1=h.get("FMeasure"), ICM=h.get("ICM"), ICMNorm=h.get("ICMNorm"),
                ICMSoft=s.get("ICMSoft"), ICMSoftNorm=s.get("ICMSoftNorm"),
                CrossEntropy=s.get("CrossEntropy"))


def compute_m1_ablation(b1, b1_base, val_tta_emb, T1, t1, val_examples, predict_test):
    """5 filas para M1 sobre validación.

    baseline : modelo M1 con target DURO y sin empates -> probs crudas, thr 0.5, sin temp, sin TTA
    +A       : modelo M1 con soft labels -> probs crudas, thr 0.5, sin temp, sin TTA
    +A+B     : ídem +A pero thr óptimo
    +A+B+C   : ídem +A+B; soft con temperature scaling
    +A+B+C+D : ídem; además TTA sobre las imágenes de validación
    """
    val_ids = b1["val_ids"]
    tgt = b1["val_targets"]
    # ---- baseline (hard target). Alinear ids del baseline a los de b1.
    idx_b = {i: k for k, i in enumerate(b1_base["val_ids"])}
    base_logits = np.array([b1_base["val_logits"][idx_b[i]] for i in val_ids])
    base_probs = 1.0 / (1.0 + np.exp(-base_logits))
    rows = [_row(val_ids, tgt, base_probs, 0.5, base_probs, "baseline")]
    # ---- M1 soft, sin TTA
    soft_logits = b1["val_logits"]
    soft_probs = 1.0 / (1.0 + np.exp(-soft_logits))
    rows.append(_row(val_ids, tgt, soft_probs, 0.5, soft_probs, "+A"))
    rows.append(_row(val_ids, tgt, soft_probs, t1, soft_probs, "+A+B"))
    cal_probs = E.apply_temperature(soft_logits, T1)
    rows.append(_row(val_ids, tgt, soft_probs, t1, cal_probs, "+A+B+C"))
    # ---- + TTA sobre validación
    pv = predict_test(b1, val_examples, val_tta_emb)          # devuelve probs calibradas (b1 tiene T)
    order = {i: k for k, i in enumerate(pv["ids"])}
    tta_logits = np.array([pv["logits_mean"][order[i]] for i in val_ids])
    tta_probs_raw = 1.0 / (1.0 + np.exp(-tta_logits))
    tta_probs_cal = E.apply_temperature(tta_logits, T1)
    rows.append(_row(val_ids, tgt, tta_probs_raw, t1, tta_probs_cal, "+A+B+C+D"))
    return rows


def calibrate_and_threshold(name, val_ids, val_logits, val_targets, results):
    """Fija temperature (Mejora C) y threshold óptimo (Mejora B) sobre validación."""
    T = E.fit_temperature(val_logits, val_targets)
    t_opt, icm_opt, _ = E.optimize_threshold(
        val_ids, 1.0 / (1.0 + np.exp(-val_logits)), val_targets)
    results["temperature"][name] = T
    results["thresholds"][name] = t_opt
    print(f"  [{name}] temperature={T:.4f}  threshold óptimo={t_opt}  (ICM val={icm_opt:.4f})")
    return T, t_opt


def main():
    print("=" * 60)
    print(f"EXIST 2026 — Task 2.1 (COMPLETO) | DRY_RUN={C.DRY_RUN} | TEAM={C.TEAM_NAME}")
    print("=" * 60)
    C.set_seed(C.SEED)
    C.configure_gpu()

    # ---- 1) datos
    print("\n[1] Datos ...")
    splits = D.load_split()
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    json.dump(splits["sensor_stats"], open(os.path.join(C.PRE_DIR, "sensor_stats.json"), "w"))

    # ---- 2) embeddings ViT estándar (train+val+test) y TTA (test)
    print("\n[2] Embeddings ViT ...")
    all_ex = splits["train"] + splits["val"] + splits["test"]
    vit_emb = precompute_vit_embeddings(all_ex)
    from inference import (precompute_test_tta_embeddings, precompute_tta_embeddings,
                           predict_test, apply_hard_decision, TTA_EMB_PATH)
    tta_emb = precompute_test_tta_embeddings(splits["test"])
    # TTA sobre validación: solo se usa para la fila +A+B+C+D de la tabla de ablación de M1
    val_tta_emb = precompute_tta_embeddings(
        splits["val"], os.path.join(C.PRE_DIR, "vit_embeddings_val_tta.npz"), tag="ViT-TTA-val")

    # ---- 3) (Gemini corre en background; las descripciones se cargan justo antes de M3) ----
    from train import train_model

    results = dict(temperature={}, thresholds={}, val_metrics={}, test_pred={}, bundles={})

    def bt_bi(key):
        bt, bi = C.BATCH[key]
        if C.DRY_RUN:
            return 8, 16
        return bt, bi

    # ============ M1 ============
    cfg_m1 = dict(text=True, image=True, et=True, hr=True, eeg=True, caption=False,
                  set_pool=False, emotions=True)
    bt, bi = bt_bi("M1")
    b1 = train_model("M1", cfg_m1, bt, bi, seed=42, splits=splits, vit_emb=vit_emb,
                     strict_phase2=C.STRICT_PHASE2)
    T1, t1 = calibrate_and_threshold("M1", b1["val_ids"], b1["val_logits"], b1["val_targets"], results)
    b1["T"] = T1
    results["bundles"]["M1"] = b1

    # --- M1 'baseline' para la ablación incremental: target DURO + descarta empates 3-3 (tipo paper Arcos)
    splits_noties = dict(splits)
    splits_noties["train"] = D.drop_ties(splits["train"])
    b1_base = train_model("M1_baseline_ablation", cfg_m1, bt, bi, seed=42,
                          splits=splits_noties, vit_emb=vit_emb, hard_target=True)

    # ============ M2 ============
    cfg_m2 = dict(text=True, image=True, et=True, hr=True, eeg=True, caption=False,
                  set_pool=True, emotions=True)
    bt, bi = bt_bi("M2")
    b2 = train_model("M2", cfg_m2, bt, bi, seed=42, splits=splits, vit_emb=vit_emb)
    T2, t2 = calibrate_and_threshold("M2", b2["val_ids"], b2["val_logits"], b2["val_targets"], results)
    b2["T"] = T2
    results["bundles"]["M2"] = b2

    # ============ M3 (vistas) ============
    # Esperar a que termine el precompute de Gemini (corre en background desde run_all.sh).
    _wait_for_gemini()
    captions = load_gemini_captions()
    use_view_E = captions is not None
    if use_view_E:
        print(f"  [Gemini] {len(captions)} descripciones cargadas para la Vista E de M3")
    else:
        print("  [aviso] sin descripciones Gemini -> M3 usará 4 vistas (A,B,C,D), voto >=3/4")
    views = ["A", "B", "C", "D"] + (["E"] if use_view_E else [])
    view_bundles = {}
    for v in views:
        vc = C.M3_VIEWS[v]
        cfg = dict(text=vc["text"], image=vc["image"], et=vc["et"], hr=vc["hr"],
                   eeg=vc["eeg"], caption=vc["caption"], set_pool=vc["set_pool"],
                   emotions=vc["emotions"])
        key = "M3_D" if v == "D" else "M3_ABCE"
        bt, bi = bt_bi(key)
        bn = f"M3_vista_{v}"
        bv = train_model(bn, cfg, bt, bi, seed=vc["seed"], splits=splits, vit_emb=vit_emb,
                         captions=captions, use_caption=vc["caption"])
        Tv, tv = calibrate_and_threshold(bn, bv["val_ids"], bv["val_logits"], bv["val_targets"], results)
        bv["T"] = Tv
        view_bundles[v] = bv
        results["bundles"][bn] = bv

    # ---- 4) métricas de validación de M1, M2, M3 (ensemble)
    print("\n[3] Métricas de validación ...")
    bl = E.baseline_metrics(b1["val_ids"], b1["val_targets"])
    for k, vv in bl.items():
        print(f"  {k}: ICM={vv.get('ICM'):.4f}  FMeasure={vv.get('FMeasure'):.4f}")

    def model_val_metrics(name, val_ids, val_logits, val_targets, T, t_opt):
        probs = 1.0 / (1.0 + np.exp(-val_logits))
        hard = E.eval_hard(val_ids, probs >= t_opt, val_targets)
        soft = E.eval_soft(val_ids, E.apply_temperature(val_logits, T), val_targets)
        m = dict(**{k: hard.get(k) for k in E.METRICS_HARD},
                 **{k: soft.get(k) for k in E.METRICS_SOFT})
        results["val_metrics"][name] = m
        print(f"  {name}: ICM={m['ICM']}  FMeasure={m['FMeasure']}  ICMSoft={m['ICMSoft']}")
        if m["ICM"] is not None and m["ICM"] <= max(bl["baseline_majority"]["ICM"],
                                                    bl["baseline_minority"]["ICM"]):
            print(f"  *** WARNING: {name} NO supera baselines en ICM ***")
        return m

    model_val_metrics("M1", b1["val_ids"], b1["val_logits"], b1["val_targets"], T1, t1)
    model_val_metrics("M2", b2["val_ids"], b2["val_logits"], b2["val_targets"], T2, t2)

    # M3 ensemble en validación (soft = media de probs -> logit -> temperature ; hard = voto mayoritario)
    # alinear por id
    base_ids = view_bundles[views[0]]["val_ids"]
    def aligned(bundle):
        idx = {i: k for k, i in enumerate(bundle["val_ids"])}
        return np.array([bundle["val_logits"][idx[i]] for i in base_ids])
    view_logits = {v: aligned(view_bundles[v]) for v in views}
    view_probs = {v: 1.0 / (1.0 + np.exp(-view_logits[v])) for v in views}
    ens_prob = np.mean(np.stack([view_probs[v] for v in views]), axis=0)
    ens_logit = _logit(ens_prob)
    val_targets_m3 = view_bundles[views[0]]["val_targets"]
    T_m3 = E.fit_temperature(ens_logit, val_targets_m3)
    results["temperature"]["M3"] = T_m3
    # hard: cada vista vota con su threshold
    need = (len(views) // 2) + 1
    votes = np.sum(np.stack([(view_probs[v] >= results["thresholds"][f"M3_vista_{v}"]).astype(int)
                             for v in views]), axis=0)
    m3_hard_pred = votes >= need
    # threshold "global" M3 lo dejamos informativo
    m3_hard = E.eval_hard(base_ids, m3_hard_pred, val_targets_m3)
    m3_soft = E.eval_soft(base_ids, E.apply_temperature(ens_logit, T_m3), val_targets_m3)
    results["val_metrics"]["M3"] = dict(**{k: m3_hard.get(k) for k in E.METRICS_HARD},
                                        **{k: m3_soft.get(k) for k in E.METRICS_SOFT})
    print(f"  M3 (ensemble {len(views)} vistas): ICM={m3_hard.get('ICM')}  "
          f"FMeasure={m3_hard.get('FMeasure')}  ICMSoft={m3_soft.get('ICMSoft')}")

    # ---- 4b) ablación incremental de M1 (5 filas: baseline, +A, +A+B, +A+B+C, +A+B+C+D)
    print("\n[3b] Ablación incremental de M1 ...")
    ablation_rows = compute_m1_ablation(b1, b1_base, val_tta_emb, T1, t1, splits["val"], predict_test)
    results["ablation_m1"] = ablation_rows
    for r in ablation_rows:
        print(f"  M1 {r['Mejoras']}: AUC={r['AUC']}  F1+={r['F1']}  ICM={r['ICM']}  ICMSoft={r['ICMSoft']}")

    # ---- 5) inferencia sobre test con TTA + reglas E/F
    print("\n[4] Inferencia sobre test (TTA) ...")
    test_ex = splits["test"]
    test_ids = [e["id"] for e in test_ex]

    pred1 = predict_test(b1, test_ex, tta_emb)
    pred2 = predict_test(b2, test_ex, tta_emb)
    pred_views = {v: predict_test(view_bundles[v], test_ex, tta_emb,
                                  captions=(captions if C.M3_VIEWS[v]["caption"] else None))
                  for v in views}

    # reordenar a test_ids
    def reorder(pred):
        idx = {i: k for k, i in enumerate(pred["ids"])}
        return np.array([pred["probs"][idx[i]] for i in test_ids])
    p1 = reorder(pred1)
    p2 = reorder(pred2)
    pv = {v: reorder(pred_views[v]) for v in views}

    # guardar attention weights (algunos ejemplos)
    for src_name, pred in [("M2", pred2)] + [(f"M3_vista_{v}", pred_views[v]) for v in views]:
        for k, (_id, aw) in enumerate(list(pred.get("attn", {}).items())[:30]):
            np.savez_compressed(os.path.join(C.ATTN_DIR, f"{src_name}_attention_{_id}.npz"),
                                **{f"alpha_{m}": aw[m] for m in aw}, image_id=_id)

    # --- Run 1 hard: M1 + A+B+D
    run1_hard = p1 >= results["thresholds"]["M1"]
    # --- Run 2 hard: M2 + A+B+D+E
    run2_hard = apply_hard_decision(test_ex, p2, results["thresholds"]["M2"],
                                    use_doubt_rule=True, train_examples=splits["train"])
    # --- Run 3 hard: M3 + A+B+D+E+F (cada vista: threshold + regla E, luego voto F)
    view_hard = {}
    for v in views:
        view_hard[v] = apply_hard_decision(test_ex, pv[v], results["thresholds"][f"M3_vista_{v}"],
                                           use_doubt_rule=True, train_examples=splits["train"])
    votes_test = np.sum(np.stack([view_hard[v].astype(int) for v in views]), axis=0)
    run3_hard = votes_test >= need

    # --- soft runs
    run1_soft = p1   # ya calibrada (T1) dentro de predict_test
    run2_soft = p2
    ens_test_prob = np.mean(np.stack([pv[v] for v in views]), axis=0)
    run3_soft = E.apply_temperature(_logit(ens_test_prob), results["temperature"]["M3"])

    # ---- 6) generar los 6 ficheros de submission
    print("\n[5] Generando submissions ...")
    sd = C.OUT_DIR
    files = {
        f"task2_1_hard_{C.TEAM_NAME}_1": ("hard", run1_hard),
        f"task2_1_hard_{C.TEAM_NAME}_2": ("hard", run2_hard),
        f"task2_1_hard_{C.TEAM_NAME}_3": ("hard", run3_hard),
        f"task2_1_soft_{C.TEAM_NAME}_1": ("soft", run1_soft),
        f"task2_1_soft_{C.TEAM_NAME}_2": ("soft", run2_soft),
        f"task2_1_soft_{C.TEAM_NAME}_3": ("soft", run3_soft),
    }
    for fname, (kind, arr) in files.items():
        path = os.path.join(sd, fname)
        if kind == "hard":
            D.write_hard_file(path, test_ids, np.asarray(arr).astype(bool))
        else:
            D.write_soft_file(path, test_ids, np.asarray(arr, dtype=float))
        print(f"  {fname}  ({len(test_ids)} preds)")

    # ---- 7) validación de formato
    print("\n[6] Validación de formato ...")
    ok = validate_files(list(files.keys()), set(test_ids))
    if not ok and not C.DRY_RUN:
        print("  *** ABORTANDO: fallos de formato ***")
        sys.exit(1)

    # ---- 8) resumen + CSVs
    write_summary_csvs(results, bl)

    # ---- 9) empaquetado del zip (solo los 6 ficheros)
    print("\n[7] Empaquetando zip ...")
    zip_path = os.path.join(C.ROOT, f"exist2026_{C.TEAM_NAME}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files:
            zf.write(os.path.join(sd, fname), arcname=os.path.join(f"exist2026_{C.TEAM_NAME}", fname))
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  {zip_path}  ({size_mb:.2f} MB)")

    print_final_summary(results, files, size_mb)


# --------------------------------------------------------------------------
def validate_files(fnames, expected_ids):
    all_ok = True
    # 1) intentar el formatting script oficial sobre una copia en carpeta task2_1/
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fmtval", C.FORMAT_VAL_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as td:
            for fn in fnames:
                shutil.copy(os.path.join(C.OUT_DIR, fn), os.path.join(td, fn))
            mod.process_format_runs_by_task(td + os.sep)
        print("  (formatting script oficial ejecutado — revisar arriba si hay 'ERROR')")
    except Exception as ex:
        print(f"  [aviso] no se pudo usar el formatting script oficial: {ex}")
    # 2) chequeo manual robusto
    for fn in fnames:
        is_soft = "soft" in fn
        try:
            manual_format_check(os.path.join(C.OUT_DIR, fn), is_soft, expected_ids)
            print(f"  OK  {fn}")
        except AssertionError as ex:
            print(f"  FALLO {fn}: {ex}")
            all_ok = False
    return all_ok


def manual_format_check(run_file, is_soft, expected_ids):
    data = json.load(open(run_file))
    assert isinstance(data, list)
    if not C.DRY_RUN:
        assert len(data) == 1053, f"len={len(data)}"
    ids = set()
    for item in data:
        assert set(item.keys()) == {"test_case", "id", "value"}
        assert item["test_case"] == "EXIST2025"
        assert isinstance(item["id"], str)
        assert item["id"] not in ids
        ids.add(item["id"])
        if is_soft:
            assert set(item["value"].keys()) == {"YES", "NO"}
            assert abs(sum(item["value"].values()) - 1.0) < 1e-4
        else:
            assert item["value"] in {"YES", "NO"}
    assert ids == expected_ids, "ids no coinciden con el test"
    return True


def write_summary_csvs(results, bl):
    # final_metrics_summary.csv
    p = os.path.join(C.VAL_DIR, "final_metrics_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sistema", "ICM", "ICMNorm", "FMeasure", "ICMSoft", "ICMSoftNorm", "CrossEntropy"])
        w.writerow(["baseline_majority", bl["baseline_majority"].get("ICM"),
                    bl["baseline_majority"].get("ICMNorm"), bl["baseline_majority"].get("FMeasure"),
                    "", "", ""])
        w.writerow(["baseline_minority", bl["baseline_minority"].get("ICM"),
                    bl["baseline_minority"].get("ICMNorm"), bl["baseline_minority"].get("FMeasure"),
                    "", "", ""])
        w.writerow(["Paper_Arcos_reportado", "N/A", "N/A", 0.722, "N/A", "N/A", "N/A"])
        for name in ("M1", "M2", "M3"):
            m = results["val_metrics"].get(name, {})
            w.writerow([name, m.get("ICM"), m.get("ICMNorm"), m.get("FMeasure"),
                        m.get("ICMSoft"), m.get("ICMSoftNorm"), m.get("CrossEntropy")])
    # thresholds / temperature
    json.dump(results["thresholds"], open(os.path.join(C.VAL_DIR, "thresholds_optimos.json"), "w"), indent=2)
    json.dump(results["temperature"], open(os.path.join(C.VAL_DIR, "temperature_scaling.json"), "w"), indent=2)
    # ablation_table.csv : ablación incremental para M1 (5 filas) + 1 fila final para M2 y M3
    pa = os.path.join(C.VAL_DIR, "ablation_table.csv")
    with open(pa, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Modelo", "Mejoras_aplicadas", "AUC", "F1_positivo", "ICM", "ICMNorm",
                    "ICMSoft", "ICMSoftNorm", "CrossEntropy"])
        for r in results.get("ablation_m1", []):
            w.writerow(["M1", r["Mejoras"], r["AUC"], r["F1"], r["ICM"], r["ICMNorm"],
                        r["ICMSoft"], r["ICMSoftNorm"], r["CrossEntropy"]])
        for name in ("M2", "M3"):
            m = results["val_metrics"].get(name, {})
            w.writerow([name, "todo (A..F segun aplique)", "", m.get("FMeasure"), m.get("ICM"),
                        m.get("ICMNorm"), m.get("ICMSoft"), m.get("ICMSoftNorm"), m.get("CrossEntropy")])
    print(f"  CSVs escritos en {C.VAL_DIR}")


def print_final_summary(results, files, size_mb):
    vm = results["val_metrics"]
    print("\n" + "=" * 60)
    print("RESUMEN DE SUBMISSION — EXIST 2026 Task 2.1")
    print("=" * 60)
    print("Modelos entrenados:")
    for name in ("M1", "M2", "M3"):
        m = vm.get(name, {})
        print(f"  M{name[-1]} — val ICMSoft: {m.get('ICMSoft')}  ICM: {m.get('ICM')}  "
              f"FMeasure: {m.get('FMeasure')}")
    best = max(("M1", "M2", "M3"), key=lambda n: (vm.get(n, {}).get("ICMSoft") or -1e18))
    print(f"\nMejor modelo: {best}")
    f1 = vm.get(best, {}).get("FMeasure")
    if f1:
        print(f"Comparación vs paper Arcos (F1=0.722): {'+' if f1>0.722 else ''}{f1-0.722:.3f} pts FMeasure")
    print("\nArchivos generados:")
    for fn in files:
        print(f"  {fn}")
    print(f"\nZIP final: exist2026_{C.TEAM_NAME}.zip ({size_mb:.2f} MB)")
    print("\nPRÓXIMO PASO: subir manualmente el ZIP a https://forms.gle/5hY91c7aBv563oZM7")
    print("Recordatorio: SOLO UNA submission por equipo.")
    print("=" * 60)


def train_M1_only_debug(epochs_p1=2, epochs_p2=5, strict_phase1=False, strict_phase2=True):
    """Test rápido de validación: entrena SOLO M1 con pocas épocas.

    Úsalo para verificar el fix antes de relanzar el pipeline completo (15-18 h).
    Activa DEBUG_FWD=1 para ver stats de features en la primera batch.
    """
    import data as D
    from precompute import precompute_vit_embeddings
    from train import train_model
    C.PHASE1_EPOCHS = epochs_p1
    C.PHASE2_EPOCHS = epochs_p2
    C.set_seed(C.SEED)
    C.configure_gpu()
    print(f"\n[DEBUG M1] Fase1={epochs_p1} épocas, Fase2={epochs_p2} épocas, "
          f"strict_phase2={strict_phase2}")
    splits = D.load_split()
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    all_ex = splits["train"] + splits["val"] + splits["test"]
    vit_emb = precompute_vit_embeddings(all_ex)
    cfg_m1 = dict(text=True, image=True, et=True, hr=True, eeg=True, caption=False,
                  set_pool=False, emotions=True)
    bt, bi = C.BATCH["M1"]
    b1 = train_model("M1", cfg_m1, bt, bi, seed=42, splits=splits, vit_emb=vit_emb,
                     strict_phase1=strict_phase1, strict_phase2=strict_phase2)
    print("\n[DEBUG M1] best_metrics:", b1["best_metrics"])
    return b1


if __name__ == "__main__":
    main()
