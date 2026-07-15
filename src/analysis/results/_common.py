"""Helpers compartidos para los bloques de análisis: gold de validación, métricas de
calibración, wrappers PyEvALL (binario / 3-clases / 5-cat) con parche sigma=0, y
extracción de Gemini con el bug 0.0->0.5 corregido."""
import os, sys, json, math, tempfile, csv, glob
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import config as C, data as D
HERE=os.path.dirname(os.path.abspath(__file__))

INT=["NO","DIRECT","JUDGEMENTAL"]
CATS=["IDEOLOGICAL-INEQUALITY","STEREOTYPING-DOMINANCE","OBJECTIFICATION","SEXUAL-VIOLENCE","MISOGYNY-NON-SEXUAL-VIOLENCE"]
CATCOLS=["P_IDEOLOGICAL","P_STEREOTYPING","P_OBJECTIFICATION","P_SEXUAL_VIOLENCE","P_MISOGYNY_NSV"]
TC="EXIST2025"

# ---- monkeypatch PyEvALL sigma=0 ----
from statistics import NormalDist
from pyevall.metrics.metrics import ICMSoft
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
def _safe(self, t, c):
    if t is None or not t[0]: return 0
    if t[0] not in self.gold_average: return -math.log2(1/len(c.gold_df))
    if t[1]==0.0: return 0.0
    sigma=max(float(self.gold_deviation[t[0]]),1e-9)
    try: prob=1-NormalDist(mu=self.gold_average[t[0]],sigma=sigma).cdf(t[1])
    except: return -math.log2(1/len(c.gold_df))
    if prob<=0.0: return -math.log2(1/len(c.gold_df))
    return -math.log2(prob)
ICMSoft.get_prob_class=_safe

_G=None
def gemini():
    global _G
    if _G is None: _G=json.load(open(os.path.join(C.PRE_DIR,"gemini_predictions.json")))
    return _G

# ---------- GOLD de validación ----------
_GOLD=None
def gold():
    """Devuelve dict con gold de los 598 IDs de validación:
       t21_soft{id->p}, t21_hard{id->0/1 (count>3)},
       t22_soft{id->[3]}, t22_hardidx{id->int},
       t23_soft{id->[5]}, t23_sex{id->p}, t23_hardcats{id->set()}  (count>1 por cat)
    """
    global _GOLD
    if _GOLD is not None: return _GOLD
    splits=D.load_split(); val_ids=[e["id"] for e in splits["val"]]
    raw=json.load(open(C.TRAIN_JSON,encoding="utf-8"))
    by_id={str(m["id_EXIST"]):m for m in raw.values()}
    t21s,t21h,t22s,t22h,t23s,t23sex,t23h={},{},{},{},{},{},{}
    for mid in val_ids:
        m=by_id[mid]
        # 2.1
        t1=[v for v in m.get("labels_task2_1",[]) if v in ("YES","NO")]; n1=len(t1) or 1
        nyes=sum(1 for v in t1 if v=="YES")
        t21s[mid]=nyes/n1; t21h[mid]=int(nyes>3)
        # 2.2 (- -> NO)
        t2=[("NO" if v=="-" else v) for v in m.get("labels_task2_2",[]) if v!="UNKNOWN"]
        c2={x:t2.count(x) for x in INT}; n2=sum(c2.values()) or 1
        t22s[mid]=[c2[x]/n2 for x in INT]; t22h[mid]=int(np.argmax([c2[x] for x in INT]))
        # 2.3
        votes=[]
        for v in m.get("labels_task2_3",[]):
            if v=="UNKNOWN": continue
            if isinstance(v,list): votes.append([x for x in v if x in CATS])
            else: votes.append([])
        n3=len(votes) or 1
        cnt={c:sum(1 for vv in votes if c in vv) for c in CATS}
        t23s[mid]=[cnt[c]/n3 for c in CATS]
        nsex=sum(1 for vv in votes if vv)
        t23sex[mid]=nsex/n3
        t23h[mid]=set(c for c in CATS if cnt[c]>1)  # >1 anotador
    _GOLD=dict(val_ids=val_ids,t21_soft=t21s,t21_hard=t21h,t22_soft=t22s,t22_hardidx=t22h,
               t23_soft=t23s,t23_sex=t23sex,t23_hardcats=t23h)
    return _GOLD

def load_preds(name):
    """Lee preds_val_<name>.csv -> (ids list, np.array [N,k])."""
    rows=list(csv.DictReader(open(os.path.join(HERE,f"preds_val_{name}.csv"))))
    cols=[c for c in rows[0].keys() if c.startswith("P_")]
    ids=[r["id_meme"] for r in rows]
    arr=np.array([[float(r[c]) for c in cols] for r in rows])
    return ids,arr

# ---------- Calibración ----------
def bin_stats(p,y,nbins=10):
    edges=np.linspace(0,1,nbins+1); out=[]
    for b in range(nbins):
        lo,hi=edges[b],edges[b+1]
        m=(p>=lo)&(p<hi) if b<nbins-1 else (p>=lo)&(p<=hi)
        if m.sum()==0: out.append((0,np.nan,np.nan)); continue
        out.append((int(m.sum()),float(p[m].mean()),float(y[m].mean())))
    return out,edges
def ece_mce(p,y,nbins=10):
    stats,_=bin_stats(p,y,nbins); n=len(p); ece=mce=0.0
    for cnt,conf,acc in stats:
        if cnt==0: continue
        gap=abs(acc-conf); ece+=(cnt/n)*gap; mce=max(mce,gap)
    return ece,mce
def brier(p,y): return float(np.mean((p-y)**2))

# ---------- PyEvALL wrappers ----------
def _run(pf,gf,metrics,hier=None):
    params={PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
            PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE}
    if hier is not None: params[PyEvALLUtils.PARAM_HIERARCHY]=hier
    rep=PyEvALLEvaluation().evaluate(pf,gf,metrics,**params)
    m=rep.report["metrics"]; out={}
    for k in metrics:
        try: out[k]=m[k]["results"]["average_per_test_case"]
        except: out[k]=None
    return out

def icm_binary(ids,pred_bool,soft_gold_hard):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":("YES" if pred_bool[k] else "NO")} for k,i in enumerate(ids)],open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":("YES" if soft_gold_hard[k] else "NO")} for k,i in enumerate(ids)],open(gf,"w"))
        return _run(pf,gf,["ICM","ICMNorm","FMeasure"])
def icmsoft_binary(ids,probs,soft_gold):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":{"YES":float(probs[k]),"NO":float(1-probs[k])}} for k,i in enumerate(ids)],open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":{"YES":float(soft_gold[k]),"NO":float(1-soft_gold[k])}} for k,i in enumerate(ids)],open(gf,"w"))
        return _run(pf,gf,["ICMSoft","ICMSoftNorm"])

HIER22={"YES":["DIRECT","JUDGEMENTAL"],"NO":[]}
def icm_22(ids,pred_idx,gold_idx,soft=False,probs=None,gold_soft=None):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        if soft:
            json.dump([{"test_case":TC,"id":str(i),"value":{INT[k]:float(probs[j,k]/max(probs[j].sum(),1e-9)) for k in range(3)}} for j,i in enumerate(ids)],open(pf,"w"))
            json.dump([{"test_case":TC,"id":str(i),"value":{INT[k]:float(gold_soft[j][k]) for k in range(3)}} for j,i in enumerate(ids)],open(gf,"w"))
            return _run(pf,gf,["ICMSoft","ICMSoftNorm"],HIER22)
        json.dump([{"test_case":TC,"id":str(i),"value":INT[int(pred_idx[j])]} for j,i in enumerate(ids)],open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":INT[int(gold_idx[j])]} for j,i in enumerate(ids)],open(gf,"w"))
        return _run(pf,gf,["ICM","ICMNorm","FMeasure"],HIER22)

HIER23={"YES":CATS,"NO":[]}
def icm_23_hard(ids,gold_cats,pred_cats):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":pred_cats[k]} for k,i in enumerate(ids)],open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":gold_cats[k]} for k,i in enumerate(ids)],open(gf,"w"))
        return _run(pf,gf,["ICM","ICMNorm","FMeasure"],HIER23)
def icm_23_soft(ids,gold_soft5,gold_sex,ps,pc):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        def s(p_no,c5): return {"NO":float(max(0.0,p_no)),**{CATS[c]:float(c5[c]) for c in range(5)}}
        json.dump([{"test_case":TC,"id":str(i),"value":s(1-ps[k],ps[k]*pc[k])} for k,i in enumerate(ids)],open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":s(1-gold_sex[k],gold_soft5[k])} for k,i in enumerate(ids)],open(gf,"w"))
        return _run(pf,gf,["ICMSoft","ICMSoftNorm"],HIER23)
