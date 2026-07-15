"""BLOQUE 2 — Análisis de errores 2.3 en los 7 checkpoints (thr tsex=0.30, tcat=0.15).
Por checkpoint: métricas por categoría + globales + matriz P(pred|gold).
Sobre gold (único): co-ocurrencia, densidad multi-etiqueta, prob media Gemini por categoría."""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _common as K
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figuras"); os.makedirs(FIG,exist_ok=True)
g=K.gold(); CATS=K.CATS
TSEX,TCAT=0.30,0.15

def pred_bin(ps,pc):
    P=np.zeros((len(ps),5),int)
    for i in range(len(ps)):
        if ps[i]<TSEX: continue
        for c in range(5):
            if pc[i,c]>=TCAT: P[i,c]=1
    return P

T23=["vista_e_task23_best","vista_e_task23_max512","vista_e_task23_max512_v2","vista_e_task23_max512_R","vista_e_task23_longformer","vista_e_task23_longformer_v2","vista_e_task23_longformer_R"]
percat=[]; confus=[]
for n in T23:
    ids,pc=K.load_preds(n); ps=pc.max(1)
    G=np.array([[1 if (g["t23_sex"][i]>=0.5 and g["t23_soft"][i][c]>1/6+1e-9) else 0 for c in range(5)] for i in ids])
    Pr=pred_bin(ps,pc)
    # global
    yb=G; pb=Pr
    f1micro=f1_score(yb.ravel(),pb.ravel())
    f1macro=np.mean([f1_score(G[:,c],Pr[:,c],zero_division=0) for c in range(5)])
    # PyEvALL ICM
    gh=[[CATS[c] for c in range(5) if G[k,c]] or ["NO"] for k in range(len(ids))]
    pr=[[CATS[c] for c in range(5) if Pr[k,c]] or [CATS[int(np.argmax(pc[k]))]] for k in range(len(ids))]
    hard=K.icm_23_hard(ids,gh,pr)
    soft=K.icm_23_soft(ids,[g["t23_soft"][i] for i in ids],[g["t23_sex"][i] for i in ids],ps,pc)
    for c in range(5):
        gc,pcb=G[:,c],Pr[:,c]
        tp=int(((gc==1)&(pcb==1)).sum());fp=int(((gc==0)&(pcb==1)).sum());fn=int(((gc==1)&(pcb==0)).sum());tn=int(((gc==0)&(pcb==0)).sum())
        percat.append(dict(checkpoint=n,categoria=CATS[c],freq_gold=int(gc.sum()),freq_pred=int(pcb.sum()),
            TP=tp,FP=fp,FN=fn,TN=tn,precision=round(precision_score(gc,pcb,zero_division=0),4),
            recall=round(recall_score(gc,pcb,zero_division=0),4),F1=round(f1_score(gc,pcb,zero_division=0),4),
            F1_macro_global=round(f1macro,4),F1_micro_global=round(f1micro,4),
            ICM=round(hard.get("ICM"),4) if hard.get("ICM") is not None else "",
            ICMSoft=round(soft.get("ICMSoft"),4) if soft.get("ICMSoft") is not None else "",
            ICMSoftNorm=round(soft.get("ICMSoftNorm"),4) if soft.get("ICMSoftNorm") is not None else ""))
    # P(pred cat | gold cat)
    for gi in range(5):
        idx=np.where(G[:,gi]==1)[0]; row={"checkpoint":n,"gold_cat":CATS[gi]}
        for pj in range(5):
            row["pred_"+CATS[pj]]=round(Pr[idx,pj].sum()/len(idx),3) if len(idx)>0 else 0.0
        confus.append(row)
    print(f"  {n:30s} F1macro={f1macro:.4f} F1micro={f1micro:.4f} ICM={hard.get('ICM'):+.4f} ICMSoft={soft.get('ICMSoft'):+.4f}")

with open(os.path.join(HERE,"errores_2_3_por_checkpoint.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(percat[0].keys())); w.writeheader(); w.writerows(percat)
with open(os.path.join(HERE,"confusion_por_checkpoint.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(confus[0].keys())); w.writeheader(); w.writerows(confus)

# ---- gold único ----
val_ids=g["val_ids"]
Ggold=np.array([[1 if (g["t23_sex"][i]>=0.5 and g["t23_soft"][i][c]>1/6+1e-9) else 0 for c in range(5)] for i in val_ids])
CO=np.zeros((5,5),int)
for row in Ggold:
    act=np.where(row==1)[0]
    for a in act:
        for b in act: CO[a,b]+=1
with open(os.path.join(HERE,"cooc_gold_2_3.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow([""]+CATS)
    for i in range(5): w.writerow([CATS[i]]+list(CO[i]))
# densidad multi-etiqueta (memes sexistas)
dens={}
for row in Ggold:
    k=int(row.sum())
    if k>0: dens[k]=dens.get(k,0)+1
with open(os.path.join(HERE,"densidad_multietiqueta_gold.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["num_categorias","num_memes"])
    for k in sorted(dens): w.writerow([k,dens[k]])
# prob media Gemini por categoria
gm=np.zeros(5); nval=0
for i in val_ids:
    v=K.gemini().get(str(i))
    if isinstance(v,dict):
        cp=(v.get("task2_3",{}) or {}).get("category_probabilities",{}) or {}
        gm+=np.array([float(cp.get(c,0.0) or 0.0) for c in CATS]); nval+=1
gm/=max(nval,1)
with open(os.path.join(HERE,"gemini_prob_media_por_categoria.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["categoria","prob_media_gemini","freq_gold"])
    for c in range(5): w.writerow([CATS[c],round(gm[c],4),int(Ggold[:,c].sum())])

# figura confusion del principal
ids,pc=K.load_preds("vista_e_task23_best"); ps=pc.max(1); Pr=pred_bin(ps,pc)
G=np.array([[1 if (g["t23_sex"][i]>=0.5 and g["t23_soft"][i][c]>1/6+1e-9) else 0 for c in range(5)] for i in ids])
CF=np.zeros((5,5))
for gi in range(5):
    idx=np.where(G[:,gi]==1)[0]
    for pj in range(5): CF[gi,pj]=Pr[idx,pj].sum()/len(idx) if len(idx)>0 else 0
fig,ax=plt.subplots(1,2,figsize=(14,6))
for a,M,ttl,fmt in [(ax[0],CO,"Co-ocurrencia GOLD","d"),(ax[1],CF,"P(pred|gold) — principal",".2f")]:
    im=a.imshow(M,cmap="Blues"); a.set_title(ttl)
    a.set_xticks(range(5)); a.set_yticks(range(5))
    a.set_xticklabels([c[:9] for c in CATS],rotation=45,ha="right",fontsize=8); a.set_yticklabels([c[:9] for c in CATS],fontsize=8)
    for i in range(5):
        for j in range(5):
            v=M[i,j]; a.text(j,i,(f"{int(v)}" if fmt=="d" else f"{v:.2f}"),ha="center",va="center",fontsize=7,color="white" if v>M.max()*0.6 else "black")
    fig.colorbar(im,ax=a,fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"confusion_2_3.png"),dpi=110); plt.close(fig)
print("BLOQUE 2 OK -> errores_2_3_por_checkpoint.csv, confusion_por_checkpoint.csv, cooc_gold_2_3.csv, densidad_multietiqueta_gold.csv, gemini_prob_media_por_categoria.csv, figuras/confusion_2_3.png")
