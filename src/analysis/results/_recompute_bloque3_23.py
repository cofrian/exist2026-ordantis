"""Recompute Bloque 3 (ablacion features numericas Gemini) para 2.3 con COMPUERTA DE SEXISMO REAL,
gold oficial y decode fijo tsex=0.30/tcat=0.15 (el optimo de las variantes en Bloque 5).
Infiere cada variante con features y con features=0, capturando ps=sigmoid(ls) y pc=sigmoid(lc).
Escribe las filas 2.3 de ablacion_gemini_todos.csv (conserva las de 2.2)."""
import os, sys, csv, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import config as C
import task23
if not hasattr(task23, "load_t23"): task23.load_t23 = task23.load_task23
import _pyevall23 as P

# ---- gold oficial (identico a Bloques 2/5) ----
Z = np.load(os.path.join(HERE, "cache_23_gate.npz"), allow_pickle=True)
gold_ids = [str(x) for x in Z["gold__ids"]]; T = Z["gold__T"]; SX = Z["gold__SX"]
gh = P.gold_hard_from_soft(T, SX)
G = np.array([[1 if P.CATS[c] in set(gh[k]) else 0 for c in range(5)] for k in range(len(gold_ids))])
pos = {m: k for k, m in enumerate(gold_ids)}
TSEX, TCAT = 0.30, np.full(5, 0.15)
LF = "markussagen/xlm-roberta-longformer-base-4096"; XL = C.TEXT_MODEL
ALT = os.path.join(C.OUT_DIR, "_alt")

VARIANTS = [
    ("XLM-R/512",              "task23_max512",        "vista_e_task23_max512_best.pt",        False),
    ("XLM-R/512 +bal",         "task23_max512_v2",     "vista_e_task23_max512_v2_best.pt",     False),
    ("XLM-R/512 +bal +reason", "task23_max512_R",      "vista_e_task23_max512_R_best.pt",      False),
    ("Longformer",             "task23_longformer",    "vista_e_task23_longformer_best.pt",    True),
    ("Longformer +bal",        "task23_longformer_v2", "vista_e_task23_longformer_v2_best.pt", True),
    ("Longformer +bal +reason","task23_longformer_R",  "vista_e_task23_longformer_R_best.pt",  True),
]

def infer(modname, ckfile, longformer, zero_feat):
    m = importlib.import_module(modname); tok = AutoTokenizer.from_pretrained(LF if longformer else XL)
    dl = DataLoader(m.DS23(m.load_t23()["val"]), batch_size=16, shuffle=False, collate_fn=m.collate(tok), num_workers=4)
    model = (getattr(m, "VistaELong23", None) or getattr(m, "VistaE23"))().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(ALT, ckfile), map_location="cpu", weights_only=False)["model_state_dict"], strict=False); model.eval()
    ids, PS, PC = [], [], []
    with torch.no_grad():
        for b in dl:
            bb = {k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero_feat and "feat" in bb: bb["feat"] = torch.zeros_like(bb["feat"])
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=True):
                ls, lc = model(bb)
            ids += b["id"]; PS.append(torch.sigmoid(ls).float().cpu().numpy().reshape(-1)); PC.append(torch.sigmoid(lc).float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    ps = np.concatenate(PS); pc = np.concatenate(PC)
    idx = [ids.index(mm) for mm in gold_ids]
    return ps[idx], pc[idx]

def metrics(ps, pc):
    pr = P.pred_from_probs(ps, pc, TSEX, TCAT)
    icm, icmn, fm = P.pyevall_hard_full(gold_ids, gh, pr)
    Pr = np.array([[1 if P.CATS[c] in set(pr[k]) else 0 for c in range(5)] for k in range(len(gold_ids))])
    fmac = np.mean([f1_score(G[:,c], Pr[:,c], zero_division=0) for c in range(5)])
    return fmac, icm

rows = []
print(f"{'config':26s} {'cond':16s} {'F1macro':>8} {'ICM':>8}", flush=True)
for label, modname, ckfile, lf in VARIANTS:
    res = {}
    for cond, zf in [("con_features", False), ("sin_features(0)", True)]:
        ps, pc = infer(modname, ckfile, lf, zf); fmac, icm = metrics(ps, pc); res[cond] = (fmac, icm)
        rows.append(dict(subtarea="2.3", checkpoint=label, condicion=cond, F1_macro=round(fmac,4), ICM=round(icm,4), decode="tsex0.30/tcat0.15 gate real, gold oficial"))
        print(f"{label:26s} {cond:16s} {fmac:8.4f} {icm:+8.4f}", flush=True)
    d_f = res["sin_features(0)"][0]-res["con_features"][0]; d_i = res["sin_features(0)"][1]-res["con_features"][1]
    print(f"    -> Delta(sin-con) F1={d_f:+.4f} ICM={d_i:+.4f}", flush=True)
rows.append(dict(subtarea="2.3", checkpoint="Vista E base (320)", condicion="N/A (no consume features numericas)", F1_macro="NA", ICM="NA", decode="NA"))

out = os.path.join(HERE, "ablacion_gemini_23_oficial.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subtarea","checkpoint","condicion","F1_macro","ICM","decode"]); w.writeheader(); w.writerows(rows)
print("-> ablacion_gemini_23_oficial.csv")
