"""BLOQUE 1 — Calibración de los 16 checkpoints (post-proceso Bloque 0).
ECE/MCE/Brier hard+soft; para 2.2/2.3 por clase OvR + macro; crudo vs calibrado
(temperature 2.1, Platt 2.2/2.3 principal); filas Gemini zero-shot; reliability PNGs.
CAVEAT: Platt/temperature se ajustan in-sample sobre validación (cotas superiores optimistas)."""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _common as K
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figuras"); os.makedirs(FIG,exist_ok=True)
g=K.gold(); ROWS=[]
def add(**k): ROWS.append(k)

def gem21(mid):
    v=K.gemini().get(str(mid))
    if not isinstance(v,dict): return 0.5
    p=v.get("task2_1",{}).get("sexist_probability"); return float(p) if p is not None else 0.5
def gem22(mid):
    v=K.gemini().get(str(mid)); ip=(v or {}).get("task2_2",{}).get("intention_probabilities",{}) if isinstance(v,dict) else {}
    if not ip: return np.array([1.,0.,0.])
    vec=np.array([float(ip.get(k)) if ip.get(k) is not None else 0.0 for k in K.INT]); s=vec.sum(); return vec/s if s>0 else np.array([1.,0.,0.])
def gem23(mid):
    v=K.gemini().get(str(mid))
    if not isinstance(v,dict): return np.zeros(5)
    cp=(v.get("task2_3",{}) or {}).get("category_probabilities",{}) or {}
    return np.array([float(cp.get(c,0.0) or 0.0) for c in K.CATS])

def rec(sub,ck,method,cls,p,yhard,ysoft):
    eh,mh=K.ece_mce(p,yhard); es,ms=K.ece_mce(p,ysoft)
    add(subtarea=sub,checkpoint=ck,metodo=method,clase=cls,N_bins=10,
        ECE_hard=round(eh,4),MCE_hard=round(mh,4),Brier_hard=round(K.brier(p,yhard),4),
        ECE_soft=round(es,4),Brier_soft=round(K.brier(p,ysoft),4))
    return eh

def fit_temp(logits,t):
    def nll(T):
        p=np.clip(expit(logits/T),1e-7,1-1e-7); return -np.mean(t*np.log(p)+(1-t)*np.log(1-p))
    return float(minimize_scalar(nll,bounds=(0.5,5),method="bounded").x)
def platt_fit(P,Y):
    out=[]
    for i in range(P.shape[1]):
        p=np.clip(P[:,i],1e-7,1-1e-7); z=np.log(p/(1-p)).reshape(-1,1)
        if len(np.unique(Y[:,i]))<2: out.append((1.,0.)); continue
        lr=LogisticRegression(C=1.0); lr.fit(z,Y[:,i]); out.append((float(lr.coef_[0,0]),float(lr.intercept_[0])))
    return out
def platt_apply(P,pr):
    c=np.zeros_like(P)
    for i,(a,b) in enumerate(pr):
        p=np.clip(P[:,i],1e-7,1-1e-7); z=np.log(p/(1-p)); c[:,i]=1/(1+np.exp(-(a*z+b)))
    return c
def rel(ax,p,y,title):
    stats,edges=K.bin_stats(p,y); cen=0.5*(edges[:-1]+edges[1:])
    ax.plot([0,1],[0,1],"--",color="gray",lw=1)
    ax.bar(cen,[s[2] if s[0]>0 else 0 for s in stats],width=0.09,alpha=0.6,edgecolor="black")
    xs=[s[1] for s in stats if s[0]>0]; ys=[s[2] for s in stats if s[0]>0]
    ax.plot(xs,ys,"o-",color="C3"); e,m=K.ece_mce(p,y)
    ax.set_title(f"{title}\nECE={e:.3f}",fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)

# ---------------- 2.1 ----------------
T21=["M3_vista_E_best","vista_e_task21_max512","vista_e_task21_max512_R","vista_e_task21_longformer","vista_e_task21_longformer_R"]
figE={}
for n in T21:
    ids,arr=K.load_preds(n); p=arr[:,0]
    ysoft=np.array([g["t21_soft"][i] for i in ids]); yhard=(ysoft>=0.5).astype(int)
    rec("2.1",n,"crudo","GLOBAL",p,yhard,ysoft)
    logit=np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7))); T=fit_temp(logit,ysoft)
    pt=expit(logit/T); rec("2.1",n,f"temperature(T={T:.3f})","GLOBAL",pt,yhard,ysoft)
    if n=="M3_vista_E_best":
        pg=np.array([gem21(i) for i in ids]); pb=0.6*p+0.4*pg
        rec("2.1","M3_vista_E_best","blend0.6/0.4Gemini","GLOBAL",pb,yhard,ysoft)
        figE=dict(ids=ids,p=p,pt=pt,pb=pb,yhard=yhard)
# Gemini zero-shot fila
ids,_=K.load_preds("M3_vista_E_best"); pg=np.array([gem21(i) for i in ids])
ysoft=np.array([g["t21_soft"][i] for i in ids]); yhard=(ysoft>=0.5).astype(int)
rec("2.1","Gemini_zeroshot","crudo","GLOBAL",pg,yhard,ysoft)
fig,ax=plt.subplots(1,3,figsize=(13,4.2))
rel(ax[0],figE["p"],figE["yhard"],"2.1 M3_vista_E crudo")
rel(ax[1],figE["pb"],figE["yhard"],"2.1 blend 0.6/0.4 Gemini")
rel(ax[2],pg,yhard,"2.1 Gemini zero-shot")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"reliability_2_1.png"),dpi=110); plt.close(fig)

# ---------------- 2.2 ----------------
T22=["vista_e_task22_best","vista_e_task22_max512","vista_e_task22_max512_R","vista_e_task22_longformer"]
for n in T22:
    ids,P=K.load_preds(n); Yh=np.eye(3)[np.array([g["t22_hardidx"][i] for i in ids])]; Ts=np.array([g["t22_soft"][i] for i in ids])
    er=[]
    for c,nm in enumerate(K.INT):
        er.append(rec("2.2",n,"crudo",nm,P[:,c],Yh[:,c],Ts[:,c]))
    add(subtarea="2.2",checkpoint=n,metodo="crudo",clase="MACRO_OvR",N_bins=10,ECE_hard=round(np.mean(er),4),MCE_hard="",Brier_hard="",ECE_soft="",Brier_soft="")
    if n=="vista_e_task22_best":
        platt=platt_fit(P,Yh); Pc=platt_apply(P,platt); ec=[]
        for c,nm in enumerate(K.INT):
            ec.append(rec("2.2",n,"platt",nm,Pc[:,c],Yh[:,c],Ts[:,c]))
        add(subtarea="2.2",checkpoint=n,metodo="platt",clase="MACRO_OvR",N_bins=10,ECE_hard=round(np.mean(ec),4),MCE_hard="",Brier_hard="",ECE_soft="",Brier_soft="")
        fig,ax=plt.subplots(2,3,figsize=(13,8))
        for c,nm in enumerate(K.INT):
            rel(ax[0,c],P[:,c],Yh[:,c],f"2.2 {nm} crudo"); rel(ax[1,c],Pc[:,c],Yh[:,c],f"2.2 {nm} Platt")
        fig.tight_layout(); fig.savefig(os.path.join(FIG,"reliability_2_2_perclass.png"),dpi=110); plt.close(fig)
# Gemini zero-shot
ids,_=K.load_preds("vista_e_task22_best"); Pg=np.array([gem22(i) for i in ids]); Yh=np.eye(3)[np.array([g["t22_hardidx"][i] for i in ids])]; Ts=np.array([g["t22_soft"][i] for i in ids])
er=[rec("2.2","Gemini_zeroshot","crudo",nm,Pg[:,c],Yh[:,c],Ts[:,c]) for c,nm in enumerate(K.INT)]
add(subtarea="2.2",checkpoint="Gemini_zeroshot",metodo="crudo",clase="MACRO_OvR",N_bins=10,ECE_hard=round(np.mean(er),4),MCE_hard="",Brier_hard="",ECE_soft="",Brier_soft="")

# ---------------- 2.3 ----------------
T23=["vista_e_task23_best","vista_e_task23_max512","vista_e_task23_max512_v2","vista_e_task23_max512_R","vista_e_task23_longformer","vista_e_task23_longformer_v2","vista_e_task23_longformer_R"]
for n in T23:
    ids,pc=K.load_preds(n)
    Yh=np.array([[1 if g["t23_soft"][i][c]>1/6+1e-9 else 0 for c in range(5)] for i in ids]); Ts=np.array([g["t23_soft"][i] for i in ids])
    er=[rec("2.3",n,"crudo",K.CATS[c],pc[:,c],Yh[:,c],Ts[:,c]) for c in range(5)]
    add(subtarea="2.3",checkpoint=n,metodo="crudo",clase="MACRO_OvR",N_bins=10,ECE_hard=round(np.mean(er),4),MCE_hard="",Brier_hard="",ECE_soft="",Brier_soft="")
    if n=="vista_e_task23_max512_v2":   # run 2.3 hard_1 entregado (reproduce Table 6) -> es el que se calibra
        platt=platt_fit(pc,Yh); pcc=platt_apply(pc,platt); ec=[rec("2.3",n,"platt",K.CATS[c],pcc[:,c],Yh[:,c],Ts[:,c]) for c in range(5)]
        add(subtarea="2.3",checkpoint=n,metodo="platt",clase="MACRO_OvR",N_bins=10,ECE_hard=round(np.mean(ec),4),MCE_hard="",Brier_hard="",ECE_soft="",Brier_soft="")
        fig,ax=plt.subplots(2,5,figsize=(20,8))
        for c in range(5):
            rel(ax[0,c],pc[:,c],Yh[:,c],f"{K.CATS[c][:10]} crudo"); rel(ax[1,c],pcc[:,c],Yh[:,c],f"{K.CATS[c][:10]} Platt")
        fig.tight_layout(); fig.savefig(os.path.join(FIG,"reliability_2_3_percat.png"),dpi=110); plt.close(fig)
ids,_=K.load_preds("vista_e_task23_best"); gpc=np.array([gem23(i) for i in ids])
Yh=np.array([[1 if g["t23_soft"][i][c]>1/6+1e-9 else 0 for c in range(5)] for i in ids]); Ts=np.array([g["t23_soft"][i] for i in ids])
er=[rec("2.3","Gemini_zeroshot","crudo",K.CATS[c],gpc[:,c],Yh[:,c],Ts[:,c]) for c in range(5)]
add(subtarea="2.3",checkpoint="Gemini_zeroshot",metodo="crudo",clase="MACRO_OvR",N_bins=10,ECE_hard=round(np.mean(er),4),MCE_hard="",Brier_hard="",ECE_soft="",Brier_soft="")

with open(os.path.join(HERE,"calibracion_todos.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["subtarea","checkpoint","metodo","clase","N_bins","ECE_hard","MCE_hard","Brier_hard","ECE_soft","Brier_soft"]); w.writeheader(); w.writerows(ROWS)
print(f"BLOQUE 1 OK: {len(ROWS)} filas -> calibracion_todos.csv + 3 reliability PNGs")
