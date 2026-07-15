"""Recompute Bloque 6 (ablacion fisiologia EEG+Ekman) para 2.3 con GOLD OFICIAL y el decode oficial
del principal (gate max-cat, tsex0.30/tcat0.20). 'con_fisiologia' debe reproducir Bloque 5:
F1 0.6750 / ICM -2.1484. Escribe ablacion_fisiologia_23_oficial.csv (solo 2.3)."""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import config as C
import task23 as T23
if not hasattr(T23, "load_t23"): T23.load_t23 = T23.load_task23
import _pyevall23 as P

Z = np.load(os.path.join(HERE, "cache_23_gate.npz"), allow_pickle=True)
gold_ids = [str(x) for x in Z["gold__ids"]]; T = Z["gold__T"]; SX = Z["gold__SX"]
gh = P.gold_hard_from_soft(T, SX)
G = np.array([[1 if P.CATS[c] in set(gh[k]) else 0 for c in range(5)] for k in range(len(gold_ids))])
TSEX, TCAT = 0.30, np.full(5, 0.20)   # optimo oficial del principal (Bloque 5)

def run23(zero):
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    dl = DataLoader(T23.DS23(T23.load_task23()["val"]), batch_size=16, shuffle=False, collate_fn=T23.collate(tok), num_workers=4)
    model = T23.VistaE23().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, "vista_e_task23_best.pt"), map_location="cpu", weights_only=False)["model_state_dict"], strict=False); model.eval()
    ids, PC = [], []
    with torch.no_grad():
        for b in dl:
            bb = {k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero: bb["eeg"] = torch.zeros_like(bb["eeg"]); bb["emotions"] = torch.zeros_like(bb["emotions"])
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=True): out = model(bb)
            ids += b["id"]; PC.append(out[0].float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    pc = np.concatenate(PC); idx = [ids.index(m) for m in gold_ids]
    return pc[idx]

rows = []
print(f"{'cond':32s} {'F1macro':>8} {'ICM':>8}", flush=True)
res = {}
for cond, z in [("con_fisiologia", False), ("sin_fisiologia(EEG+Ekman=0)", True)]:
    pc = run23(z); ps = pc.max(1)
    pr = P.pred_from_probs(ps, pc, TSEX, TCAT)
    icm, icmn, fm = P.pyevall_hard_full(gold_ids, gh, pr)
    Pr = np.array([[1 if P.CATS[c] in set(pr[k]) else 0 for c in range(5)] for k in range(len(gold_ids))])
    fmac = np.mean([f1_score(G[:,c], Pr[:,c], zero_division=0) for c in range(5)])
    res[cond] = (fmac, icm)
    rows.append(dict(subtarea="2.3", checkpoint="Vista E base (320)", condicion=cond, F1_macro=round(fmac,4), ICM=round(icm,4), decode="tsex0.30/tcat0.20 gate max-cat, gold oficial"))
    print(f"{cond:32s} {fmac:8.4f} {icm:+8.4f}", flush=True)
d_f = res["sin_fisiologia(EEG+Ekman=0)"][0]-res["con_fisiologia"][0]; d_i = res["sin_fisiologia(EEG+Ekman=0)"][1]-res["con_fisiologia"][1]
print(f"-> Delta(sin-con) F1={d_f:+.4f} ICM={d_i:+.4f}", flush=True)

with open(os.path.join(HERE, "ablacion_fisiologia_23_oficial.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subtarea","checkpoint","condicion","F1_macro","ICM","decode"]); w.writeheader(); w.writerows(rows)
print("-> ablacion_fisiologia_23_oficial.csv")
