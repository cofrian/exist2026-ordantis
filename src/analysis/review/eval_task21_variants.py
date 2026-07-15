"""Evalua en VALIDACION los 4 checkpoints alternativos de Task 2.1 (binario) de _alt/,
SIN reentrenar: reutiliza la clase de modelo, el DS/collate y el infer de cada modulo.
Metricas: F1+ (positiva), AUC, ICM (a threshold optimo sobre ICM), ICMSoft. Formato CSV.
"""
import os, sys, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as C, data as D
from dataset import MemeDataset, make_collate

MODULES = [
    ("Vista E-2.1 max512",         "task21_max512",       "vista_e_task21_max512_best.pt",       "base"),
    ("Vista E-2.1 max512_R",       "task21_max512_R",     "vista_e_task21_max512_R_best.pt",     "base"),
    ("Vista E-2.1 Longformer",     "task21_longformer",   "vista_e_task21_longformer_best.pt",   "long"),
    ("Vista E-2.1 Longformer_R",   "task21_longformer_R", "vista_e_task21_longformer_R_best.pt", "long"),
]

def opt_thr_icm(mod, ids, probs, y_soft):
    best = (0.5, -1e9)
    for t in [round(0.30 + 0.01*i, 2) for i in range(41)]:
        icm = mod.icm_hard(ids, (y_soft >= 0.5).astype(int), (probs >= t).astype(int))
        if icm is not None and icm > best[1]: best = (t, icm)
    return best

def eval_one(label, module_name, ckpt_name, kind):
    mod = importlib.import_module(module_name)
    splits = D.load_split()
    caps = mod.load_caps()
    ck = os.path.join(C.OUT_DIR, "_alt", ckpt_name)
    if kind == "base":
        tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
        C.MAX_TOKENS = mod.MAX_TOK
        vit = {e["id"]: np.zeros(768, np.float32) for e in splits["val"]}
        ds = MemeDataset(splits["val"], tok, vit_emb=vit, captions=caps, use_caption=True)
        dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=make_collate(tok), num_workers=4)
        model = mod.VistaE21().to(C.DEVICE)
    else:
        tok = AutoTokenizer.from_pretrained(mod.LONG_MODEL)
        ds = mod.DS21(splits["val"], caps) if "caps" in mod.DS21.__init__.__code__.co_varnames else mod.DS21(splits["val"])
        dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=mod.collate(tok), num_workers=4)
        model = mod.VistaELong21().to(C.DEVICE)
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    ids, logits, soft = mod.infer(model, dl)
    del model; torch.cuda.empty_cache()
    probs = 1/(1+np.exp(-logits))
    y = (soft >= 0.5).astype(int)
    thr, icm = opt_thr_icm(mod, ids, probs, soft)
    pred = (probs >= thr).astype(int)
    f1p = f1_score(y, pred)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    icms = mod.icmsoft(ids, soft, probs)
    print(f"  {label:26s} F1+={f1p:.4f}  AUC={auc:.4f}  ICM={icm:+.4f}  ICMSoft={icms:+.4f}  thr={thr:.2f}", flush=True)
    return dict(subtask="2.1", model=label, checkpoint=ckpt_name, F1_pos=f1p, AUC=auc,
                ICM=icm, ICMSoft=icms, thr=thr, n_val=len(ids))

if __name__ == "__main__":
    import csv
    print("=== Task 2.1 — variantes _alt (validacion) ===")
    rows = []
    for label, m, ck, kind in MODULES:
        p = os.path.join(C.OUT_DIR, "_alt", ck)
        if not os.path.exists(p):
            print(f"  [no encontrado: {ck}]"); continue
        try:
            rows.append(eval_one(label, m, ck, kind))
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  ERROR {label}: {ex}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task21_variants.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("CSV:", out)
