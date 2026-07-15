"""BLOQUE 6 — Ablación de fisiología (EEG 256 + Ekman 7 a cero) en los 3 checkpoints
principales (uno por subtarea). F1/AUC/ICM sobre validación."""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _common as K
import config as C, data as D
from dataset import MemeDataset, make_collate, to_device
HERE=os.path.dirname(os.path.abspath(__file__)); g=K.gold(); rows=[]

def caps():
    c={}
    for mid,v in K.gemini().items():
        if isinstance(v,dict):
            d=(v.get("description") or "").strip(); a=(v.get("sexism_analysis") or "").strip(); parts=[]
            if d: parts.append("Description: "+d)
            if a: parts.append("Sexism Analysis: "+a)
            if parts: c[str(mid)]=" ".join(parts)
    return c

# ---- 2.1 M3_vista_E ----
def run21(zero):
    from models import MemeClassifier
    splits=D.load_split(); C.MAX_TOKENS=256
    tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    mcfg=dict(text=True,image=False,et=False,hr=False,eeg=True,caption=True,set_pool=True,emotions=True)
    model=MemeClassifier(mcfg); model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR,"M3_vista_E_best.pt"),map_location="cpu",weights_only=False)["model_state_dict"]); model.to(C.DEVICE).eval()
    ex=splits["val"]; dl=DataLoader(MemeDataset(ex,tok,vit_emb={e["id"]:np.zeros(768,np.float32) for e in ex},captions=caps(),use_caption=True),batch_size=64,shuffle=False,collate_fn=make_collate(tok))
    ids,lo=[],[]
    with torch.no_grad():
        for b in dl:
            b=to_device(b,C.DEVICE)
            if zero:
                b["sens_EEG"]=torch.zeros_like(b["sens_EEG"]); b["emotions"]=torch.zeros_like(b["emotions"])
            with torch.autocast("cuda",dtype=C.AMP_DTYPE,enabled=True): l=model(b)
            ids+=b["id"]; lo.append(l.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return ids,1/(1+np.exp(-np.concatenate(lo)))

for cond,z in [("con_fisiologia",False),("sin_fisiologia(EEG+Ekman=0)",True)]:
    ids,p=run21(z); ys=np.array([g["t21_soft"][i] for i in ids]); yh=(ys>=0.5).astype(int)
    best=(0.5,-1e9)
    for t in [round(0.30+0.01*i,2) for i in range(41)]:
        icm=K.icm_binary(ids,(p>=t),yh).get("ICM")
        if icm is not None and icm>best[1]: best=(t,icm)
    pred=(p>=best[0]); f1=f1_score(yh,pred); auc=roc_auc_score(yh,p)
    rows.append(dict(subtarea="2.1",checkpoint="M3_vista_E_best",condicion=cond,F1=round(f1,4),AUC=round(auc,4),ICM=round(best[1],4),ICMSoft=""))
    print(f"  2.1 {cond:30s} F1+={f1:.4f} AUC={auc:.4f} ICM={best[1]:+.4f}")

# ---- 2.2 task22_best ----
import task22 as T22
def run22(zero):
    splits=T22.load_task22(); tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    dl=DataLoader(T22.DS22(splits["val"],tok),batch_size=64,shuffle=False,collate_fn=T22.collate(tok),num_workers=4)
    model=T22.VistaE22().to(C.DEVICE); model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR,"vista_e_task22_best.pt"),map_location="cpu",weights_only=False)["model_state_dict"],strict=False); model.eval()
    ids,P=[],[]
    with torch.no_grad():
        for b in dl:
            bb={k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero: bb["eeg"]=torch.zeros_like(bb["eeg"]); bb["emotions"]=torch.zeros_like(bb["emotions"])
            with torch.autocast("cuda",dtype=C.AMP_DTYPE,enabled=True): probs,_,_=model(bb)
            ids+=b["id"]; P.append(probs.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return ids,np.concatenate(P)
for cond,z in [("con_fisiologia",False),("sin_fisiologia(EEG+Ekman=0)",True)]:
    ids,P=run22(z); gi=np.array([g["t22_hardidx"][i] for i in ids])
    best=(-1,0.3,0.4)
    for tj in np.arange(0.10,0.45,0.025):
        for td in np.arange(0.25,0.60,0.025):
            pr=np.where(P[:,2]>tj,2,np.where(P[:,1]>td,1,0)); s=f1_score(gi,pr,average="macro",labels=[0,1,2])
            if s>best[0]: best=(s,tj,td)
    pr=np.where(P[:,2]>best[1],2,np.where(P[:,1]>best[2],1,0)); fm=f1_score(gi,pr,average="macro",labels=[0,1,2]); icm=K.icm_22(ids,pr,gi).get("ICM")
    rows.append(dict(subtarea="2.2",checkpoint="vista_e_task22_best",condicion=cond,F1=round(fm,4),AUC="",ICM=round(icm,4) if icm is not None else "",ICMSoft=""))
    print(f"  2.2 {cond:30s} F1macro={fm:.4f} ICM={icm:+.4f}")

# ---- 2.3 task23_best ----
import task23 as T23
def run23(zero):
    splits=T23.load_task23(); tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    dl=DataLoader(T23.DS23(splits["val"]),batch_size=16,shuffle=False,collate_fn=T23.collate(tok),num_workers=4)
    model=T23.VistaE23().to(C.DEVICE); model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR,"vista_e_task23_best.pt"),map_location="cpu",weights_only=False)["model_state_dict"],strict=False); model.eval()
    ids,PC=[],[]
    with torch.no_grad():
        for b in dl:
            bb={k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero: bb["eeg"]=torch.zeros_like(bb["eeg"]); bb["emotions"]=torch.zeros_like(bb["emotions"])
            with torch.autocast("cuda",dtype=C.AMP_DTYPE,enabled=True): out=model(bb)
            ids+=b["id"]; PC.append(out[0].float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return ids,np.concatenate(PC)
for cond,z in [("con_fisiologia",False),("sin_fisiologia(EEG+Ekman=0)",True)]:
    ids,pc=run23(z); ps=pc.max(1)
    gh=[[K.CATS[c] for c in range(5) if (g["t23_sex"][i]>=0.5 and g["t23_soft"][i][c]>1/6+1e-9)] or ["NO"] for i in ids]
    pr=[]
    for k in range(len(ids)):
        if ps[k]<0.30: pr.append(["NO"]); continue
        cs=[K.CATS[c] for c in range(5) if pc[k,c]>=0.15]; pr.append(cs if cs else [K.CATS[int(np.argmax(pc[k]))]])
    h=K.icm_23_hard(ids,gh,pr)
    rows.append(dict(subtarea="2.3",checkpoint="vista_e_task23_best",condicion=cond,F1=round(h.get("FMeasure"),4) if h.get("FMeasure") is not None else "",AUC="",ICM=round(h.get("ICM"),4) if h.get("ICM") is not None else "",ICMSoft=""))
    print(f"  2.3 {cond:30s} F1={h.get('FMeasure'):.4f} ICM={h.get('ICM'):+.4f}")

with open(os.path.join(HERE,"ablacion_fisiologia_principales.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["subtarea","checkpoint","condicion","F1","AUC","ICM","ICMSoft"]); w.writeheader(); w.writerows(rows)
print("-> ablacion_fisiologia_principales.csv")
