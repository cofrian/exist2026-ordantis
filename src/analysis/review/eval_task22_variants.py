"""Evalua en VALIDACION los 3 checkpoints alternativos de Task 2.2 (NO/DIRECT/JUDGEMENTAL)
de _alt/, SIN reentrenar. Reutiliza modelo/DS/collate/infer de cada modulo.
Metricas: F1macro (argmax), F1 por clase, ICM jerarquico (hard, argmax) e ICMSoft (soft)."""
import os, sys, json, tempfile, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as C
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

INT = ["NO", "DIRECT", "JUDGEMENTAL"]
HIER = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
TC = "EXIST2025"

MODULES = [
    ("Vista E-2.2 max512",       "task22_max512",     "vista_e_task22_max512_best.pt",     "base", "VistaE22"),
    ("Vista E-2.2 max512_R",     "task22_max512_R",   "vista_e_task22_max512_R_best.pt",   "base", "VistaE22"),
    ("Vista E-2.2 Longformer",   "task22_longformer", "vista_e_task22_longformer_best.pt", "long", "VistaELong"),
]

def pyevall_icm(ids, y, pred_idx_or_probs, soft=False):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td, "p"), os.path.join(td, "g")
        gold, preds = [], []
        for i, mid in enumerate(ids):
            if soft:
                gold.append({"test_case": TC, "id": str(mid), "value": {INT[k]: float(y[i, k]) for k in range(3)}})
                p = pred_idx_or_probs[i]; p = p / max(p.sum(), 1e-9)
                preds.append({"test_case": TC, "id": str(mid), "value": {INT[k]: float(p[k]) for k in range(3)}})
            else:
                gold.append({"test_case": TC, "id": str(mid), "value": INT[int(np.argmax(y[i]))]})
                preds.append({"test_case": TC, "id": str(mid), "value": INT[int(pred_idx_or_probs[i])]})
        json.dump(preds, open(pf, "w")); json.dump(gold, open(gf, "w"))
        metrics = ["ICMSoft", "ICMSoftNorm"] if soft else ["ICM", "ICMNorm", "FMeasure"]
        rep = PyEvALLEvaluation().evaluate(pf, gf, metrics,
            **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
               PyEvALLUtils.PARAM_HIERARCHY: HIER,
               PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        m = rep.report["metrics"]
        return {k: m[k]["results"]["average_per_test_case"] for k in metrics if k in m and "results" in m[k]}

def thr2d(probs, y):
    yy = np.argmax(y, 1); best = (-1, 0.30, 0.40)
    for tj in np.arange(0.10, 0.45, 0.025):
        for td in np.arange(0.25, 0.60, 0.025):
            pred = np.where(probs[:, 2] > tj, 2, np.where(probs[:, 1] > td, 1, 0))
            s = f1_score(yy, pred, average="macro", labels=[0, 1, 2])
            if s > best[0]: best = (s, tj, td)
    return best[1], best[2]

def eval_one(label, module_name, ckpt_name, kind, model_cls):
    mod = importlib.import_module(module_name)
    splits = mod.load_t22()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL if kind == "base" else mod.LONG_MODEL)
    dl = DataLoader(mod.DS22(splits["val"]), batch_size=32, shuffle=False,
                    collate_fn=mod.collate(tok), num_workers=4)
    model = getattr(mod, model_cls)().to(C.DEVICE)
    ck = os.path.join(C.OUT_DIR, "_alt", ckpt_name)
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    ids, P, T = mod.infer(model, dl)
    del model; torch.cuda.empty_cache()
    y = np.argmax(T, 1)
    # argmax
    pred_arg = np.argmax(P, 1)
    f1m = f1_score(y, pred_arg, average="macro", labels=[0, 1, 2])
    f1c = f1_score(y, pred_arg, average=None, labels=[0, 1, 2])
    # threshold 2D optimo
    tj, td = thr2d(P, T)
    pred_thr = np.where(P[:, 2] > tj, 2, np.where(P[:, 1] > td, 1, 0))
    f1m_thr = f1_score(y, pred_thr, average="macro", labels=[0, 1, 2])
    icm_h = pyevall_icm(ids, T, pred_thr, soft=False)
    icm_s = pyevall_icm(ids, T, P, soft=True)
    print(f"  {label:24s} F1macro(argmax)={f1m:.4f} F1macro(thr)={f1m_thr:.4f} "
          f"F1[N/D/J]={f1c[0]:.2f}/{f1c[1]:.2f}/{f1c[2]:.2f} "
          f"ICM={icm_h.get('ICM',float('nan')):+.4f} ICMSoft={icm_s.get('ICMSoft',float('nan')):+.4f}", flush=True)
    return dict(subtask="2.2", model=label, checkpoint=ckpt_name,
                F1macro_argmax=f1m, F1macro_thr=f1m_thr,
                F1_NO=f1c[0], F1_DIRECT=f1c[1], F1_JUDG=f1c[2],
                ICM=icm_h.get("ICM"), ICMNorm=icm_h.get("ICMNorm"),
                ICMSoft=icm_s.get("ICMSoft"), ICMSoftNorm=icm_s.get("ICMSoftNorm"),
                thr_j=tj, thr_d=td, n_val=len(ids))

if __name__ == "__main__":
    import csv
    print("=== Task 2.2 — variantes _alt (validacion) ===")
    rows = []
    for label, m, ck, kind, cls in MODULES:
        p = os.path.join(C.OUT_DIR, "_alt", ck)
        if not os.path.exists(p):
            print(f"  [no encontrado: {ck}]"); continue
        try:
            rows.append(eval_one(label, m, ck, kind, cls))
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"  ERROR {label}: {ex}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task22_variants.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("CSV:", out)
