"""BLOQUE 3 — Ablación de las features numéricas de Gemini (nueva inferencia).
2.2 (4 ckpts): pone a 0 las 7 features (key 'gfeat' en el principal, 'feat' en variantes).
2.3 (6 variantes): pone a 0 las 6 features (key 'feat'). El principal task23_best NO consume
features numéricas -> N/A (se anota, no se fuerza)."""
import os, sys, csv, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _common as K
import config as C
HERE=os.path.dirname(os.path.abspath(__file__)); g=K.gold()
rows=[]

def infer22(module_name, ckpt, longformer, main, zero_key):
    m=importlib.import_module(module_name)
    splits=m.load_task22() if main else m.load_t22()
    tok=AutoTokenizer.from_pretrained(m.LONG_MODEL if longformer else C.TEXT_MODEL)
    ds=m.DS22(splits["val"],tok) if main else m.DS22(splits["val"])
    dl=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
    cls=getattr(m,"VistaELong",None) or getattr(m,"VistaE22")
    model=cls().to(C.DEVICE)
    ckpath=os.path.join(C.CKPT_DIR,ckpt) if main else os.path.join(C.OUT_DIR,"_alt",ckpt)
    model.load_state_dict(torch.load(ckpath,map_location="cpu",weights_only=False)["model_state_dict"],strict=False); model.eval()
    ids,P=[],[]
    with torch.no_grad():
        for b in dl:
            bb={k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero_key and zero_key in bb: bb[zero_key]=torch.zeros_like(bb[zero_key])
            with torch.autocast("cuda",dtype=C.AMP_DTYPE,enabled=True):
                probs,_,_=model(bb)
            ids+=b["id"]; P.append(probs.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return ids,np.concatenate(P)

def metrics22(ids,P):
    gi=np.array([g["t22_hardidx"][i] for i in ids]); gs=[g["t22_soft"][i] for i in ids]
    best=(-1,0.3,0.4)
    for tj in np.arange(0.10,0.45,0.025):
        for td in np.arange(0.25,0.60,0.025):
            pr=np.where(P[:,2]>tj,2,np.where(P[:,1]>td,1,0)); s=f1_score(gi,pr,average="macro",labels=[0,1,2])
            if s>best[0]: best=(s,tj,td)
    pr=np.where(P[:,2]>best[1],2,np.where(P[:,1]>best[2],1,0))
    fm=f1_score(gi,pr,average="macro",labels=[0,1,2]); fc=f1_score(gi,pr,average=None,labels=[0,1,2])
    h=K.icm_22(ids,pr,gi); s=K.icm_22(ids,None,None,soft=True,probs=P,gold_soft=gs)
    return fm,fc,h.get("ICM"),s.get("ICMSoft"),s.get("ICMSoftNorm")

print("=== 2.2 ablación gfeat/feat ===")
T22=[("vista_e_task22_best","task22","vista_e_task22_best.pt",False,True,"gfeat"),
     ("vista_e_task22_max512","task22_max512","vista_e_task22_max512_best.pt",False,False,"feat"),
     ("vista_e_task22_max512_R","task22_max512_R","vista_e_task22_max512_R_best.pt",False,False,"feat"),
     ("vista_e_task22_longformer","task22_longformer","vista_e_task22_longformer_best.pt",True,False,"feat")]
for name,mod,ck,lf,mn,key in T22:
    for cond,zk in [("con_features",None),("sin_features(0)",key)]:
        ids,P=infer22(mod,ck,lf,mn,zk); fm,fc,icm,icms,icmsn=metrics22(ids,P)
        rows.append(dict(subtarea="2.2",checkpoint=name,condicion=cond,F1_macro=round(fm,4),
            F1_por_clase=f"{fc[0]:.3f}/{fc[1]:.3f}/{fc[2]:.3f}",ICM=round(icm,4) if icm is not None else "",
            ICM_Soft=round(icms,4) if icms is not None else "",ICM_SoftNorm=round(icmsn,4) if icmsn is not None else ""))
        print(f"  {name:28s} {cond:16s} F1m={fm:.4f} ICM={icm:+.4f} ICMSoft={icms:+.4f}")

def infer23(module_name, ckpt, longformer, zero_feat):
    m=importlib.import_module(module_name); splits=m.load_t23()
    tok=AutoTokenizer.from_pretrained(m.LONG_MODEL if longformer else C.TEXT_MODEL)
    dl=DataLoader(m.DS23(splits["val"]),batch_size=16,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
    cls=getattr(m,"VistaELong23",None) or getattr(m,"VistaE23")
    model=cls().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt",ckpt),map_location="cpu",weights_only=False)["model_state_dict"],strict=False); model.eval()
    ids,PC=[],[]
    with torch.no_grad():
        for b in dl:
            bb={k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero_feat and "feat" in bb: bb["feat"]=torch.zeros_like(bb["feat"])
            with torch.autocast("cuda",dtype=C.AMP_DTYPE,enabled=True):
                ls,lc=model(bb)
            ids+=b["id"]; PC.append(torch.sigmoid(lc).float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return ids,np.concatenate(PC)

def metrics23(ids,pc):
    ps=pc.max(1)
    gh=[[K.CATS[c] for c in range(5) if (g["t23_sex"][i]>=0.5 and g["t23_soft"][i][c]>1/6+1e-9)] or ["NO"] for i in ids]
    def pr_(tsex,tcat):
        out=[]
        for k in range(len(ids)):
            if ps[k]<tsex: out.append(["NO"]); continue
            cs=[K.CATS[c] for c in range(5) if pc[k,c]>=tcat]; out.append(cs if cs else [K.CATS[int(np.argmax(pc[k]))]])
        return out
    pr=pr_(0.30,0.15); h=K.icm_23_hard(ids,gh,pr)
    s=K.icm_23_soft(ids,[g["t23_soft"][i] for i in ids],[g["t23_sex"][i] for i in ids],ps,pc)
    G=np.array([[1 if (g["t23_sex"][i]>=0.5 and g["t23_soft"][i][c]>1/6+1e-9) else 0 for c in range(5)] for i in ids])
    Pr=np.array([[1 if (ps[k]>=0.30 and pc[k,c]>=0.15) else 0 for c in range(5)] for k in range(len(ids))])
    fmac=np.mean([f1_score(G[:,c],Pr[:,c],zero_division=0) for c in range(5)])
    fc=[f1_score(G[:,c],Pr[:,c],zero_division=0) for c in range(5)]
    return fmac,fc,h.get("FMeasure"),h.get("ICM"),s.get("ICMSoft"),s.get("ICMSoftNorm")

print("=== 2.3 ablación feat (6 variantes; principal N/A) ===")
T23=[("vista_e_task23_max512","task23_max512","vista_e_task23_max512_best.pt",False),
     ("vista_e_task23_max512_v2","task23_max512_v2","vista_e_task23_max512_v2_best.pt",False),
     ("vista_e_task23_max512_R","task23_max512_R","vista_e_task23_max512_R_best.pt",False),
     ("vista_e_task23_longformer","task23_longformer","vista_e_task23_longformer_best.pt",True),
     ("vista_e_task23_longformer_v2","task23_longformer_v2","vista_e_task23_longformer_v2_best.pt",True),
     ("vista_e_task23_longformer_R","task23_longformer_R","vista_e_task23_longformer_R_best.pt",True)]
for name,mod,ck,lf in T23:
    for cond,zf in [("con_features",False),("sin_features(0)",True)]:
        ids,pc=infer23(mod,ck,lf,zf); fmac,fc,fm_pyeval,icm,icms,icmsn=metrics23(ids,pc)
        rows.append(dict(subtarea="2.3",checkpoint=name,condicion=cond,F1_macro=round(fmac,4),
            F1_por_clase="/".join(f"{x:.2f}" for x in fc),ICM=round(icm,4) if icm is not None else "",
            ICM_Soft=round(icms,4) if icms is not None else "",ICM_SoftNorm=round(icmsn,4) if icmsn is not None else ""))
        print(f"  {name:30s} {cond:16s} F1m={fmac:.4f} ICM={icm:+.4f} ICMSoft={icms:+.4f}")
rows.append(dict(subtarea="2.3",checkpoint="vista_e_task23_best",condicion="N/A (no consume features numericas de Gemini)",F1_macro="NA",F1_por_clase="NA",ICM="NA",ICM_Soft="NA",ICM_SoftNorm="NA"))

with open(os.path.join(HERE,"ablacion_gemini_todos.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["subtarea","checkpoint","condicion","F1_macro","F1_por_clase","ICM","ICM_Soft","ICM_SoftNorm"]); w.writeheader(); w.writerows(rows)
print("-> ablacion_gemini_todos.csv")
