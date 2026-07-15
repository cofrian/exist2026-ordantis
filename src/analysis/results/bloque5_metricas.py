"""BLOQUE 5 — Rendimiento oficial (PyEvALL) sobre validación de los 16 checkpoints,
+ blends (2.1/2.2) + Gemini zero-shot en las 3 subtareas. Post-proceso del Bloque 0."""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _common as K
from sklearn.metrics import f1_score, roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); g=K.gold()

def gem21(mid):
    v=K.gemini().get(str(mid))
    if not isinstance(v,dict): return 0.5
    p=v.get("task2_1",{}).get("sexist_probability"); return float(p) if p is not None else 0.5
def gem22(mid):
    v=K.gemini().get(str(mid))
    ip=(v or {}).get("task2_2",{}).get("intention_probabilities",{}) if isinstance(v,dict) else {}
    if not ip: return np.array([1.,0.,0.])
    vec=np.array([float(ip.get(k)) if ip.get(k) is not None else 0.0 for k in K.INT]); s=vec.sum()
    return vec/s if s>0 else np.array([1.,0.,0.])
def gem23(mid):
    v=K.gemini().get(str(mid))
    if not isinstance(v,dict): return 0.0,np.zeros(5)
    ps=v.get("task2_1",{}).get("sexist_probability"); ps=float(ps) if ps is not None else 0.0
    cp=(v.get("task2_3",{}) or {}).get("category_probabilities",{}) or {}
    return ps,np.array([float(cp.get(c,0.0) or 0.0) for c in K.CATS])

rows=[]

# ---------------- 2.1 ----------------
def eval21(label,ids,p):
    ysoft=np.array([g["t21_soft"][i] for i in ids]); yhard=(ysoft>=0.5).astype(int)
    # threshold optimo sobre ICM
    best=(0.5,-1e9)
    for t in [round(0.30+0.01*i,2) for i in range(41)]:
        icm=K.icm_binary(ids,(p>=t),yhard).get("ICM")
        if icm is not None and icm>best[1]: best=(t,icm)
    thr,icm=best; pred=(p>=thr)
    f1=f1_score(yhard,pred); auc=roc_auc_score(yhard,p) if len(set(yhard))>1 else float("nan")
    icms=K.icmsoft_binary(ids,p,ysoft).get("ICMSoft")
    rows.append(dict(subtarea="2.1",sistema=label,metrica_F1="F1_pos",F1=round(f1,4),F1_extra=f"AUC={auc:.4f}",
                     ICM=round(icm,4) if icm else "",ICMSoft=round(icms,4) if icms is not None else "",thr=thr))
    print(f"  2.1 {label:34s} F1+={f1:.4f} AUC={auc:.4f} ICM={icm:+.4f} ICMSoft={icms:+.4f} thr={thr}")

T21=["M3_vista_E_best","vista_e_task21_max512","vista_e_task21_max512_R","vista_e_task21_longformer","vista_e_task21_longformer_R"]
main_p=None
for n in T21:
    ids,arr=K.load_preds(n); p=arr[:,0]
    if n=="M3_vista_E_best": main_p=(ids,p)
    eval21(n,ids,p)
# blend principal + Gemini
ids,p=main_p; pg=np.array([gem21(i) for i in ids])
eval21("blend 0.6*M3_vista_E+0.4*Gemini",ids,0.6*p+0.4*pg)
eval21("Gemini zero-shot",ids,pg)

# ---------------- 2.2 ----------------
def eval22(label,ids,P):
    goldidx=np.array([g["t22_hardidx"][i] for i in ids]); goldsoft=[g["t22_soft"][i] for i in ids]
    # argmax
    pa=np.argmax(P,1)
    f1a=f1_score(goldidx,pa,average="macro",labels=[0,1,2])
    ha=K.icm_22(ids,pa,goldidx)
    # threshold 2D optimo
    best=(-1,0.3,0.4)
    for tj in np.arange(0.10,0.45,0.025):
        for td in np.arange(0.25,0.60,0.025):
            pr=np.where(P[:,2]>tj,2,np.where(P[:,1]>td,1,0))
            s=f1_score(goldidx,pr,average="macro",labels=[0,1,2])
            if s>best[0]: best=(s,tj,td)
    prt=np.where(P[:,2]>best[1],2,np.where(P[:,1]>best[2],1,0))
    ht=K.icm_22(ids,prt,goldidx); f1t=f1_score(goldidx,prt,average="macro",labels=[0,1,2])
    f1c=f1_score(goldidx,prt,average=None,labels=[0,1,2])
    hs=K.icm_22(ids,None,None,soft=True,probs=P,gold_soft=goldsoft)
    rows.append(dict(subtarea="2.2",sistema=label,metrica_F1="F1_macro",F1=round(f1t,4),
                     F1_extra=f"argmax={f1a:.4f};N/D/J={f1c[0]:.2f}/{f1c[1]:.2f}/{f1c[2]:.2f}",
                     ICM=round(ht.get('ICM'),4) if ht.get('ICM') is not None else "",
                     ICMSoft=round(hs.get('ICMSoft'),4) if hs.get('ICMSoft') is not None else "",thr=f"{best[1]:.2f}/{best[2]:.2f}"))
    print(f"  2.2 {label:34s} F1m(thr)={f1t:.4f} F1m(arg)={f1a:.4f} ICM={ht.get('ICM'):+.4f} ICMSoft={hs.get('ICMSoft'):+.4f}")

T22=["vista_e_task22_best","vista_e_task22_max512","vista_e_task22_max512_R","vista_e_task22_longformer"]
main22=None
for n in T22:
    ids,P=K.load_preds(n)
    if n=="vista_e_task22_best": main22=(ids,P)
    eval22(n,ids,P)
ids,P=main22; Pg=np.array([gem22(i) for i in ids]); bl=0.6*P+0.4*Pg; bl=bl/bl.sum(1,keepdims=True)
eval22("blend 0.6*task22_best+0.4*Gemini",ids,bl)
eval22("Gemini zero-shot",ids,Pg)

# ---------------- 2.3 ---------------- (thresholds tsex=0.30, tcat=0.15)
def pred_cats(ps,pc,tsex=0.30,tcat=0.15):
    out=[]
    for i in range(len(ps)):
        if ps[i]<tsex: out.append(["NO"]); continue
        cs=[K.CATS[c] for c in range(5) if pc[i,c]>=tcat]
        out.append(cs if cs else [K.CATS[int(np.argmax(pc[i]))]])
    return out
def gold_cats23(ids):
    out=[]
    for i in ids:
        if g["t23_sex"][i]<0.5: out.append(["NO"]); continue
        s=g["t23_soft"][i]; cs=[K.CATS[c] for c in range(5) if s[c]>1/6+1e-9]
        out.append(cs if cs else [K.CATS[int(np.argmax(s))]])
    return out
def eval23(label,ids,pc,ps):
    yb=np.array([[1 if g["t23_soft"][i][c]>1/6+1e-9 else 0 for c in range(5)] for i in ids])
    pb=(pc>=0.5).astype(int)*(np.array([g["t23_sex"][i] for i in ids])[:,None]>=0.5)
    f1micro=f1_score(yb.ravel(),pb.ravel())
    gh=gold_cats23(ids); pr=pred_cats(ps,pc)
    h=K.icm_23_hard(ids,gh,pr); fm=h.get("FMeasure")
    soft5=[g["t23_soft"][i] for i in ids]; sex=[g["t23_sex"][i] for i in ids]
    s=K.icm_23_soft(ids,soft5,sex,ps,pc)
    rows.append(dict(subtarea="2.3",sistema=label,metrica_F1="F1_macro",F1=round(fm,4) if fm is not None else "",
                     F1_extra=f"F1micro={f1micro:.4f}",ICM=round(h.get('ICM'),4) if h.get('ICM') is not None else "",
                     ICMSoft=round(s.get('ICMSoft'),4) if s.get('ICMSoft') is not None else "",thr="tsex0.30/tcat0.15"))
    print(f"  2.3 {label:34s} F1macro={fm:.4f} F1micro={f1micro:.4f} ICM={h.get('ICM'):+.4f} ICMSoft={s.get('ICMSoft'):+.4f}")

T23=["vista_e_task23_best","vista_e_task23_max512","vista_e_task23_max512_v2","vista_e_task23_max512_R",
     "vista_e_task23_longformer","vista_e_task23_longformer_v2","vista_e_task23_longformer_R"]
for n in T23:
    ids,pc=K.load_preds(n); ps=pc.max(1)  # compuerta de sexista = max prob categoría
    eval23(n,ids,pc,ps)
# Gemini zero-shot 2.3
ids,_=K.load_preds("vista_e_task23_best"); gps=np.array([gem23(i)[0] for i in ids]); gpc=np.array([gem23(i)[1] for i in ids])
eval23("Gemini zero-shot",ids,gpc,gps)

with open(os.path.join(HERE,"metricas_todos_checkpoints_val.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["subtarea","sistema","metrica_F1","F1","F1_extra","ICM","ICMSoft","thr"]); w.writeheader(); w.writerows(rows)
print("-> metricas_todos_checkpoints_val.csv")
