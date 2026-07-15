"""Recomputa Bloques 2 (errores) y 5 (metricas) para Task 2.3 con la COMPUERTA DE SEXISMO REAL.
Usa la cache cache_23_gate.npz (ya inferida) y helpers PyEvALL COPIADOS de _full_eval_task23.py
(NO se importa ese modulo porque su codigo a nivel de __main__ re-infiere todo y se cuelga).
Sin GPU, sin re-inferencia. Valida vs oficial; escribe CSV solo con --write."""
import os, sys, csv, json, tempfile, math
import numpy as np
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__))
WRITE = "--write" in sys.argv

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

CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]
HIER = {"YES": CATS, "NO": []}; TC = "EXIST2025"

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

# ---------------------------------------------------------------- datos
Z = np.load(os.path.join(HERE, "cache_23_gate.npz"), allow_pickle=True)
gold_ids = [str(x) for x in Z["gold__ids"]]; T = Z["gold__T"]; SX = Z["gold__SX"]
gh = gold_hard_from_soft(T, SX)
G = np.array([[1 if CATS[c] in set(gh[k]) else 0 for c in range(5)] for k in range(len(gold_ids))])

ORDER = ["vista_e_task23_best","vista_e_task23_max512","vista_e_task23_max512_v2",
         "vista_e_task23_max512_R","vista_e_task23_longformer","vista_e_task23_longformer_v2",
         "vista_e_task23_longformer_R"]

def aligned(name):
    ids = [str(x) for x in Z[f"{name}__ids"]]; ps = Z[f"{name}__ps"]; pc = Z[f"{name}__pc"]
    idx = [ids.index(m) for m in gold_ids]
    return ps[idx], pc[idx]

met_rows, err_rows, conf_rows = [], [], []
OFICIAL = {"vista_e_task23_max512_v2": (0.7146, 0.3417), "vista_e_task23_max512": (0.7137, 0.3363)}
print(f"{'checkpoint':32s} {'tsex':>5} {'tcat':>5} {'Fmacro':>7} {'ICM':>8} {'Fmicro':>7} {'ICMSoft':>9}  chk", flush=True)
for name in ORDER:
    ps, pc = aligned(name)
    tsex, tcat = find_best_thr(gold_ids, ps, pc, T, SX)
    pr = pred_from_probs(ps, pc, tsex, tcat)
    icm, icmn, fm = pyevall_hard_full(gold_ids, gh, pr)
    icms, icmsn = pyevall_soft_full(gold_ids, T, SX, ps, pc)
    Pr = np.array([[1 if CATS[c] in set(pr[k]) else 0 for c in range(5)] for k in range(len(gold_ids))])
    fmicro = f1_score(G.ravel(), Pr.ravel())
    tag = f"tsex{tsex:.2f}/tcat{tcat[0]:.2f}"
    met_rows.append(["2.3", name, "F1_macro", f"{fm:.4f}", f"F1micro={fmicro:.4f}", f"{icm:.4f}", f"{icms:.4f}", tag])
    f1s = []
    for c in range(5):
        tp=int(((Pr[:,c]==1)&(G[:,c]==1)).sum()); fp=int(((Pr[:,c]==1)&(G[:,c]==0)).sum())
        fn=int(((Pr[:,c]==0)&(G[:,c]==1)).sum()); tn=int(((Pr[:,c]==0)&(G[:,c]==0)).sum())
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9); f1s.append(f1)
        err_rows.append([name,CATS[c],int(G[:,c].sum()),int(Pr[:,c].sum()),tp,fp,fn,tn,
                         f"{prec:.4f}",f"{rec:.4f}",f"{f1:.4f}","","","","",""])
    base=len(err_rows)-5
    err_rows[base][11]=f"{np.mean(f1s):.4f}"; err_rows[base][12]=f"{fmicro:.4f}"
    err_rows[base][13]=f"{icm:.4f}"; err_rows[base][14]=f"{icms:.4f}"; err_rows[base][15]=f"{icmsn:.4f}"
    for c in range(5):
        mask=G[:,c]==1; denom=max(int(mask.sum()),1)
        conf_rows.append([name,CATS[c]]+[f"{Pr[mask,cc].sum()/denom:.3f}" for cc in range(5)])
    chk=""
    if name in OFICIAL:
        ef,ei=OFICIAL[name]; chk="OK-oficial" if (abs(fm-ef)<0.01 and abs(icm-ei)<0.01) else "<<DISCREPA"
    print(f"{name:32s} {tsex:5.2f} {tcat[0]:5.2f} {fm:7.4f} {icm:+8.4f} {fmicro:7.4f} {icms:+9.4f}  {chk}", flush=True)

if not WRITE:
    print("\n[dry-run] revisa OK-oficial; usa --write para sobrescribir los CSV"); sys.exit(0)

mp=os.path.join(HERE,"metricas_todos_checkpoints_val.csv")
with open(mp) as f: allrows=list(csv.reader(f))
head=allrows[0]; keep=[r for r in allrows[1:] if r and r[0]!="2.3"]
gem=[r for r in allrows[1:] if r and r[0]=="2.3" and "Gemini" in r[1]]
with open(mp,"w",newline="") as f:
    w=csv.writer(f); w.writerow(head)
    for r in keep: w.writerow(r)
    for r in met_rows: w.writerow(r)
    for r in gem: w.writerow(r)
print("-> metricas_todos_checkpoints_val.csv (filas 2.3 actualizadas; Gemini 2.3 conservado)")
with open(os.path.join(HERE,"errores_2_3_por_checkpoint.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["checkpoint","categoria","freq_gold","freq_pred","TP","FP","FN","TN",
        "precision","recall","F1","F1_macro_global","F1_micro_global","ICM","ICMSoft","ICMSoftNorm"]); w.writerows(err_rows)
print("-> errores_2_3_por_checkpoint.csv")
with open(os.path.join(HERE,"confusion_por_checkpoint.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["checkpoint","gold_cat"]+[f"pred_{c}" for c in CATS]); w.writerows(conf_rows)
print("-> confusion_por_checkpoint.csv")
