"""Tabla de métricas alineada a los 18 RUNS reales de la submission (6 por subtarea),
según el diseño descrito por el equipo. Reutiliza las métricas ya calculadas (validación
n=598) y RELLENA los huecos: Platt por clase de las variantes 2.2 (soft_2/soft_3).
Los blends con receta no confirmada (2.1 soft_1 'GEMF'; 2.2 hard_3/soft_1) se marcan PENDIENTE.
"""
import os, sys, json, tempfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); HERE = os.path.dirname(os.path.abspath(__file__))
import config as C
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

INT = ["NO","DIRECT","JUDGEMENTAL"]; HIER22 = {"YES":["DIRECT","JUDGEMENTAL"],"NO":[]}; TC="EXIST2025"

def platt(P, Y):
    cal = np.zeros_like(P)
    for i in range(P.shape[1]):
        p = np.clip(P[:,i],1e-7,1-1e-7); z=np.log(p/(1-p)).reshape(-1,1)
        if len(np.unique(Y[:,i]))<2: cal[:,i]=P[:,i]; continue
        lr=LogisticRegression(C=1.0); lr.fit(z,Y[:,i]); cal[:,i]=lr.predict_proba(z)[:,1]
    return cal

def icmsoft_22(ids, T, probs):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        gold=[{"test_case":TC,"id":str(m),"value":{INT[k]:float(T[i,k]) for k in range(3)}} for i,m in enumerate(ids)]
        pr=[]
        for i,m in enumerate(ids):
            p=probs[i]; p=p/max(p.sum(),1e-9)
            pr.append({"test_case":TC,"id":str(m),"value":{INT[k]:float(p[k]) for k in range(3)}})
        json.dump(pr,open(pf,"w")); json.dump(gold,open(gf,"w"))
        rep=PyEvALLEvaluation().evaluate(pf,gf,["ICMSoft","ICMSoftNorm"],
            **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
               PyEvALLUtils.PARAM_HIERARCHY:HIER22,PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        m=rep.report["metrics"]; return m["ICMSoft"]["results"]["average_per_test_case"], m["ICMSoftNorm"]["results"]["average_per_test_case"]

def platt_soft_22(module_name, ckpt, longformer):
    import importlib; mod=importlib.import_module(module_name)
    splits=mod.load_t22(); tok=AutoTokenizer.from_pretrained(mod.LONG_MODEL if longformer else C.TEXT_MODEL)
    dl=DataLoader(mod.DS22(splits["val"]),batch_size=32,shuffle=False,collate_fn=mod.collate(tok),num_workers=4)
    model=getattr(mod,"VistaELong" if longformer else "VistaE22")().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt",ckpt),map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,P,T=mod.infer(model,dl); del model; torch.cuda.empty_cache()
    Y=np.eye(3)[np.argmax(T,1)]; Pc=platt(P,Y)
    return icmsoft_22(ids,T,Pc)

# -------- cargar métricas ya calculadas de los CSV --------
import csv
def load_csv(name):
    p=os.path.join(HERE,name); return list(csv.DictReader(open(p))) if os.path.exists(p) else []
t21=load_csv("task21_variants.csv"); t22=load_csv("task22_variants.csv")
def find(rows,ck):
    for r in rows:
        if r["checkpoint"]==ck: return r
    return {}

# 2.3 desde metricas_18_modelos
m18=load_csv("metricas_18_modelos.csv")
def find23(ck):
    for r in m18:
        if r["subtask"]=="2.3" and r["checkpoint"]==ck: return r
    return {}

print("Calculando Platt soft de variantes 2.2 (soft_2, soft_3)...", flush=True)
try: ls2_ce, ls2_cen = platt_soft_22("task22_longformer","vista_e_task22_longformer_best.pt", True)
except Exception as e: ls2_ce=None; print("  err longformer:",e)
try: ls3_ce, ls3_cen = platt_soft_22("task22_max512","vista_e_task22_max512_best.pt", False)
except Exception as e: ls3_ce=None; print("  err max512:",e)

rows=[]
def R(sub,run,cfg,ck,f1,icm,icmsoft,nota=""):
    rows.append(dict(subtarea=sub,run=run,config=cfg,checkpoint=ck,F1=f1,ICM=icm,ICMSoft=icmsoft,nota=nota))

# ---------------- 2.1 ----------------
# blends ya calculados en validación (blends_val.py)
_B = json.load(open(os.path.join(HERE,"_blends_val.json"))) if os.path.exists(os.path.join(HERE,"_blends_val.json")) else {}
r=find(t21,"vista_e_task21_max512_R_best.pt");    R("2.1","hard_1","XLM-R 512 + reasoning","vista_e_task21_max512_R", r.get("F1_pos"), r.get("ICM"), r.get("ICMSoft"))
r=find(t21,"vista_e_task21_longformer_best.pt");  R("2.1","hard_2","Longformer 1100 (sin reasoning)","vista_e_task21_longformer", r.get("F1_pos"), r.get("ICM"), r.get("ICMSoft"))
r=find(t21,"vista_e_task21_max512_best.pt");      R("2.1","hard_3","XLM-R 512 (sin reasoning)","vista_e_task21_max512", r.get("F1_pos"), r.get("ICM"), r.get("ICMSoft"))
_b=_B.get("t21_soft1",{}); R("2.1","soft_1","0.6·P(max512_R) + 0.4·P(Gemini sexist_prob)","vista_e_task21_max512_R + Gemini", _b.get("F1"), (f"{_b.get('ICM'):+.4f}" if _b.get('ICM') is not None else ""), (f"{_b.get('ICMSoft'):+.4f}" if _b.get('ICMSoft') is not None else ""), "GEMF = max512_R (confirmado)")
r=find(t21,"vista_e_task21_longformer_best.pt");  R("2.1","soft_2","Longformer 1100 (sin reasoning), prob directa","vista_e_task21_longformer", r.get("F1_pos"), r.get("ICM"), r.get("ICMSoft"))
r=find(t21,"vista_e_task21_longformer_R_best.pt");R("2.1","soft_3","Longformer 1100 + reasoning, prob directa","vista_e_task21_longformer_R", r.get("F1_pos"), r.get("ICM"), r.get("ICMSoft"))

# ---------------- 2.2 ----------------
r=find(t22,"vista_e_task22_longformer_best.pt");  R("2.2","hard_1","Longformer 1100, thresholds jerárquicos","vista_e_task22_longformer", r.get("F1macro_thr"), r.get("ICM"), r.get("ICMSoft"))
r=find(t22,"vista_e_task22_max512_best.pt");      R("2.2","hard_2","XLM-R 512, thresholds jerárquicos","vista_e_task22_max512", r.get("F1macro_thr"), r.get("ICM"), r.get("ICMSoft"))
_b=_B.get("t22_hard3",{}); R("2.2","hard_3","blend 0.6/0.4 solo DIRECT/JUDG, P(NO)=1-(D+J), argmax","vista_e_task22_longformer + Gemini", _b.get("F1"), (f"{_b.get('ICM'):+.4f}" if _b.get('ICM') is not None else ""), "", "blend confirmado")
_b=_B.get("t22_soft1",{}); R("2.2","soft_1","blend 0.6/0.4 por clase (renormalizado)","vista_e_task22_longformer + Gemini", _b.get("F1"), "", (f"{_b.get('ICMSoft'):+.4f}" if _b.get('ICMSoft') is not None else ""), "blend confirmado")
R("2.2","soft_2","Longformer 1100 + Platt por clase","vista_e_task22_longformer", "", "", (f"{ls2_ce:+.4f}" if ls2_ce is not None else "err"), "ICMSoft con Platt (nuevo cálculo)")
R("2.2","soft_3","XLM-R 512 + Platt por clase","vista_e_task22_max512", "", "", (f"{ls3_ce:+.4f}" if ls3_ce is not None else "err"), "ICMSoft con Platt (nuevo cálculo)")

# ---------------- 2.3 ----------------
def g23(ck): r=find23(ck); return r.get("F1"), r.get("ICM"), r.get("ICMSoft"), r.get("extra","")
f,i,s,ex=g23("vista_e_task23_max512_v2_best.pt");    R("2.3","hard_1","XLM-R 512 + sampler","vista_e_task23_max512_v2", f,i,s, ex.split(';')[0])
f,i,s,ex=g23("vista_e_task23_max512_best.pt");       R("2.3","hard_2","XLM-R 512 sin sampler","vista_e_task23_max512", f,i,s, ex.split(';')[0])
f,i,s,ex=g23("vista_e_task23_max512_R_best.pt");     R("2.3","hard_3","XLM-R 512 + sampler + reasoning","vista_e_task23_max512_R", f,i,s, ex.split(';')[0])
f,i,s,ex=g23("vista_e_task23_longformer_best.pt");   R("2.3","soft_1","Longformer 1100 sin sampler","vista_e_task23_longformer", f,i,s, ex.split(';')[0])
f,i,s,ex=g23("vista_e_task23_max512_best.pt");       R("2.3","soft_2","XLM-R 512 sin sampler","vista_e_task23_max512", f,i,s, ex.split(';')[0])
f,i,s,ex=g23("vista_e_task23_longformer_v2_best.pt");R("2.3","soft_3","Longformer 1100 + sampler","vista_e_task23_longformer_v2", f,i,s, ex.split(';')[0])

out=os.path.join(HERE,"metricas_por_run.csv")
with open(out,"w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=["subtarea","run","config","checkpoint","F1","ICM","ICMSoft","nota"]); w.writeheader(); w.writerows(rows)
print("\n== metricas_por_run.csv ==")
for r in rows:
    print(f"  {r['subtarea']} {r['run']:7s} {r['checkpoint']:32s} F1={r['F1'] or '—':<7} ICM={r['ICM'] or '—':<8} ICMSoft={r['ICMSoft'] or '—':<9} {r['nota']}")
print("\nCSV:", out)
