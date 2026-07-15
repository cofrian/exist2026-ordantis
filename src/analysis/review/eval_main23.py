"""Evaluacion del checkpoint 2.3 PRINCIPAL (vista_e_task23_best.pt, el del zip).

Este modelo (task23.VistaE23) SOLO produce 5 sigmoides de categoria; no tiene
cabeza de 'sexista'. Por eso task23.infer devuelve (ids, P[5], T[5], D) y el
harness generico _full_eval_task23.py (escrito para las variantes con 6 salidas)
no lo desempaqueta. Aqui reutilizamos EXACTAMENTE los helpers PyEvALL de
_full_eval_task23.py con una compuerta de sexista = max prob de categoria.

Salidas:
  - imprime F1micro / F1macro / ICM / ICMSoft / ICMSoftNorm (mismo formato que items 2-7)
  - guarda resultados_revision/cache_main23_val.npz con ids, P (probs cat), T (gold soft cat),
    sex_real, para reutilizar en TAREA 3 (errores) y TAREA 4 sin recomputar.
"""
import json, os, tempfile, math, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as C, data as D
import task23 as T23

# ---- monkeypatch PyEvALL sigma=0 (identico a _full_eval_task23.py) ----
from statistics import NormalDist
from pyevall.metrics.metrics import ICMSoft
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
def _safe(self, t, c):
    if t is None or not t[0]: return 0
    if t[0] not in self.gold_average: return -math.log2(1/len(c.gold_df))
    if t[1] == 0.0: return 0.0
    sigma = max(float(self.gold_deviation[t[0]]), 1e-9)
    try: prob = 1 - NormalDist(mu=self.gold_average[t[0]], sigma=sigma).cdf(t[1])
    except: return -math.log2(1/len(c.gold_df))
    if prob <= 0.0: return -math.log2(1/len(c.gold_df))
    return -math.log2(prob)
ICMSoft.get_prob_class = _safe

CATS = T23.CATS
ALL = ["NO"] + CATS; HIER = {"YES": CATS, "NO": []}
TC = "EXIST2025"

def gold_hard_from_soft(soft5, sex):
    y = []
    for i in range(len(sex)):
        if sex[i] < 0.5: y.append(["NO"]); continue
        cats = [CATS[c] for c in range(5) if soft5[i, c] > (1.0/6 + 1e-9)]
        y.append(cats if cats else [CATS[int(np.argmax(soft5[i]))]])
    return y

def pred_from_probs(ps, pc, thr_sex=0.5, thr_cat=None):
    if thr_cat is None: thr_cat = np.full(5, 0.5)
    out = []
    for i in range(len(ps)):
        if ps[i] < thr_sex: out.append(["NO"]); continue
        cats = [CATS[c] for c in range(5) if pc[i, c] >= thr_cat[c]]
        out.append(cats if cats else [CATS[int(np.argmax(pc[i]))]])
    return out

def pyevall_hard_full(ids, gold, pred):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td,"p"), os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":pred[k]} for k,i in enumerate(ids)], open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":gold[k]} for k,i in enumerate(ids)], open(gf,"w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICM","ICMNorm","FMeasure"],
            **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
               PyEvALLUtils.PARAM_HIERARCHY:HIER,
               PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        m = rep.report["metrics"]
        return (m["ICM"]["results"]["average_per_test_case"],
                m["ICMNorm"]["results"]["average_per_test_case"],
                m["FMeasure"]["results"]["test_cases"][0]["average"])

def pyevall_soft_full(ids, soft5, sex, ps, pc):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td,"p"), os.path.join(td,"g")
        def s(p_no, c5): return {"NO": float(max(0.0,p_no)), **{CATS[c]:float(c5[c]) for c in range(5)}}
        json.dump([{"test_case":TC,"id":str(i),"value":s(1-ps[k], ps[k]*pc[k])} for k,i in enumerate(ids)], open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":s(1-sex[k], soft5[k])} for k,i in enumerate(ids)], open(gf,"w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICMSoft","ICMSoftNorm"],
            **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
               PyEvALLUtils.PARAM_HIERARCHY:HIER,
               PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        m = rep.report["metrics"]
        return (m["ICMSoft"]["results"]["average_per_test_case"],
                m["ICMSoftNorm"]["results"]["average_per_test_case"])

def find_best_thr(vids, vPS, vPC, vT, vSX):
    gh = gold_hard_from_soft(vT, vSX); best=(-1,0.5,np.full(5,0.5))
    for tsex in np.arange(0.30, 0.66, 0.04):
        for tc in np.arange(0.05, 0.55, 0.05):
            pr = pred_from_probs(vPS, vPC, tsex, np.full(5, tc))
            icm, icmn, fm = pyevall_hard_full(vids, gh, pr)
            sc = 0.5*fm + 0.5*icmn
            if sc > best[0]: best = (sc, float(tsex), np.full(5, float(tc)))
    return best[1], best[2]

def main():
    DEV = C.DEVICE
    splits = T23.load_task23()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    cl = T23.collate(tok)
    model = T23.VistaE23().to(DEV)
    ck = os.path.join(C.CKPT_DIR, "vista_e_task23_best.pt")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    dl_va = DataLoader(T23.DS23(splits["val"]), batch_size=16, shuffle=False, collate_fn=cl, num_workers=4)
    ids, P, Tt, Dz = T23.infer(model, dl_va)   # P: prob cat [N,5], Tt: soft cat gold [N,5]
    del model; torch.cuda.empty_cache()

    # gold sexista real desde labels_task2_1 (voto mayoritario), como en _full_eval item 8
    raw = json.load(open(C.TRAIN_JSON, encoding="utf-8"))
    sex_real = {}
    for m in raw.values():
        t1 = [v for v in m.get("labels_task2_1", []) if v in ("YES","NO")]
        n = len(t1) or 1
        sex_real[str(m["id_EXIST"])] = sum(1 for v in t1 if v == "YES")/n
    SX = np.array([sex_real[i] for i in ids])

    # compuerta de sexista para el modelo principal (no tiene cabeza de sexista):
    #   ps = max_c prob_categoria  (sexista <=> confia en alguna categoria)
    ps = P.max(axis=1)
    pc = P

    # 1) F1 micro cat @0.5 (con compuerta sexista real >=0.5, identico a items 2-7)
    yb = (Tt > 1/6 + 1e-9).astype(int)
    pb = (pc >= 0.5).astype(int) * (SX[:, None] >= 0.5)
    f1micro = f1_score(yb.ravel(), pb.ravel())
    # 2) thr optimo + F1macro + ICM hard
    tsex, tcat = find_best_thr(ids, ps, pc, Tt, SX)
    gh = gold_hard_from_soft(Tt, SX)
    pr = pred_from_probs(ps, pc, tsex, tcat)
    icm, icmn, fm = pyevall_hard_full(ids, gh, pr)
    # 3) ICMSoft
    icmsoft, icmsoftn = pyevall_soft_full(ids, Tt, SX, ps, pc)

    print("=== 1. Vista E-2.3 ORIGINAL (zip, max=320, sampler) [evaluador dedicado] ===")
    print(f"  F1 micro (cat @ 0.5):      {f1micro:.4f}")
    print(f"  thr_sex={tsex:.2f}  thr_cat={tcat[0]:.2f}")
    print(f"  F1macro (con thr):         {fm:.4f}")
    print(f"  ICM hard (con thr):        {icm:+.4f}    ICMNorm: {icmn:.4f}")
    print(f"  ICMSoft:                   {icmsoft:+.4f}    ICMSoftNorm: {icmsoftn:.4f}")

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_main23_val.npz"),
             ids=np.array(ids), P=P, T=Tt, SX=SX, tsex=tsex, tcat=tcat)
    print("  [cache guardado: cache_main23_val.npz]")

if __name__ == "__main__":
    main()
