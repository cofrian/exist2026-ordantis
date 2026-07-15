"""BLOQUE 0 — Predicciones crudas (sin threshold) de los 16 checkpoints sobre VALIDACION (598).
Un CSV por checkpoint. Usa la interfaz confirmada de cada modelo (la que generó la submission real).
  2.1: id_meme, P_YES
  2.2: id_meme, P_NO, P_DIRECT, P_JUDG
  2.3: id_meme, P_IDEOLOGICAL, P_STEREOTYPING, P_OBJECTIFICATION, P_SEXUAL_VIOLENCE, P_MISOGYNY_NSV
"""
import os, sys, json, importlib, warnings, csv
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import config as C, data as D
from dataset import MemeDataset, make_collate, to_device
OUT=os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT,exist_ok=True)
G=json.load(open(os.path.join(C.PRE_DIR,"gemini_predictions.json")))

def caps_desc_analysis():
    caps={}
    for mid,v in G.items():
        if isinstance(v,dict):
            d=(v.get("description") or "").strip(); a=(v.get("sexism_analysis") or "").strip()
            parts=[]
            if d: parts.append("Description: "+d)
            if a: parts.append("Sexism Analysis: "+a)
            if parts: caps[str(mid)]=" ".join(parts)
    return caps

def write_csv(name, header, rows):
    p=os.path.join(OUT,f"preds_val_{name}.csv")
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  [OK] preds_val_{name}.csv  ({len(rows)} filas)")

# ==================== TASK 2.1 ====================
def t21_main():
    from models import MemeClassifier
    splits=D.load_split(); caps=caps_desc_analysis()
    C.MAX_TOKENS=256   # M3_vista_E entrenado a 256
    tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    mcfg=dict(text=True,image=False,et=False,hr=False,eeg=True,caption=True,set_pool=True,emotions=True)
    model=MemeClassifier(mcfg)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR,"M3_vista_E_best.pt"),map_location="cpu",weights_only=False)["model_state_dict"])
    model.to(C.DEVICE).eval()
    ex=splits["val"]
    dl=DataLoader(MemeDataset(ex,tok,vit_emb={e["id"]:np.zeros(768,np.float32) for e in ex},captions=caps,use_caption=True),
                  batch_size=64,shuffle=False,collate_fn=make_collate(tok))
    ids,lo=[],[]
    with torch.no_grad():
        for b in dl:
            b=to_device(b,C.DEVICE)
            with torch.autocast("cuda",dtype=C.AMP_DTYPE,enabled=True): l=model(b)
            ids+=b["id"]; lo.append(l.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    p=1/(1+np.exp(-np.concatenate(lo)))
    write_csv("M3_vista_E_best",["id_meme","P_YES"],[[i,f"{pp:.6f}"] for i,pp in zip(ids,p)])

def t21_variant(name, module_name, ckpt, kind):
    m=importlib.import_module(module_name); splits=D.load_split(); caps=m.load_caps()
    if kind=="base":
        tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL); C.MAX_TOKENS=m.MAX_TOK
        vit={e["id"]:np.zeros(768,np.float32) for e in splits["val"]}
        dl=DataLoader(MemeDataset(splits["val"],tok,vit_emb=vit,captions=caps,use_caption=True),
                      batch_size=32,shuffle=False,collate_fn=make_collate(tok),num_workers=4)
        model=m.VistaE21().to(C.DEVICE)
    else:
        tok=AutoTokenizer.from_pretrained(m.LONG_MODEL)
        dl=DataLoader(m.DS21(splits["val"],caps),batch_size=8,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
        model=m.VistaELong21().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt",ckpt),map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,lo,_=m.infer(model,dl); del model; torch.cuda.empty_cache()
    p=1/(1+np.exp(-lo))
    write_csv(name,["id_meme","P_YES"],[[i,f"{pp:.6f}"] for i,pp in zip(ids,p)])

# ==================== TASK 2.2 ====================
def t22(name, module_name, ckpt, longformer, main=False):
    m=importlib.import_module(module_name)
    splits=m.load_task22() if main else m.load_t22()
    tok=AutoTokenizer.from_pretrained(m.LONG_MODEL if longformer else C.TEXT_MODEL)
    ds=m.DS22(splits["val"],tok) if main else m.DS22(splits["val"])
    dl=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
    cls=getattr(m,"VistaELong",None) or getattr(m,"VistaE22")
    model=cls().to(C.DEVICE)
    ckpath=os.path.join(C.CKPT_DIR,ckpt) if main else os.path.join(C.OUT_DIR,"_alt",ckpt)
    model.load_state_dict(torch.load(ckpath,map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,P,_=m.infer(model,dl); del model; torch.cuda.empty_cache()
    write_csv(name,["id_meme","P_NO","P_DIRECT","P_JUDG"],[[i,f"{P[k,0]:.6f}",f"{P[k,1]:.6f}",f"{P[k,2]:.6f}"] for k,i in enumerate(ids)])

# ==================== TASK 2.3 ====================
CATCOLS=["P_IDEOLOGICAL","P_STEREOTYPING","P_OBJECTIFICATION","P_SEXUAL_VIOLENCE","P_MISOGYNY_NSV"]
def t23(name, module_name, ckpt, longformer, main=False):
    m=importlib.import_module(module_name)
    splits=m.load_task23() if main else m.load_t23()
    tok=AutoTokenizer.from_pretrained(m.LONG_MODEL if longformer else C.TEXT_MODEL)
    dl=DataLoader(m.DS23(splits["val"]),batch_size=16,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
    cls=getattr(m,"VistaELong23",None) or getattr(m,"VistaE23")
    model=cls().to(C.DEVICE)
    ckpath=os.path.join(C.CKPT_DIR,ckpt) if main else os.path.join(C.OUT_DIR,"_alt",ckpt)
    model.load_state_dict(torch.load(ckpath,map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    out=m.infer(model,dl); del model; torch.cuda.empty_cache()
    ids=out[0]; pc=out[2] if len(out)>=5 else out[1]
    write_csv(name,["id_meme"]+CATCOLS,[[i]+[f"{pc[k,j]:.6f}" for j in range(5)] for k,i in enumerate(ids)])

if __name__=="__main__":
    print("=== TASK 2.1 ===")
    t21_main()
    t21_variant("vista_e_task21_max512","task21_max512","vista_e_task21_max512_best.pt","base")
    t21_variant("vista_e_task21_max512_R","task21_max512_R","vista_e_task21_max512_R_best.pt","base")
    t21_variant("vista_e_task21_longformer","task21_longformer","vista_e_task21_longformer_best.pt","long")
    t21_variant("vista_e_task21_longformer_R","task21_longformer_R","vista_e_task21_longformer_R_best.pt","long")
    print("=== TASK 2.2 ===")
    t22("vista_e_task22_best","task22","vista_e_task22_best.pt",False,main=True)
    t22("vista_e_task22_max512","task22_max512","vista_e_task22_max512_best.pt",False)
    t22("vista_e_task22_max512_R","task22_max512_R","vista_e_task22_max512_R_best.pt",False)
    t22("vista_e_task22_longformer","task22_longformer","vista_e_task22_longformer_best.pt",True)
    print("=== TASK 2.3 ===")
    t23("vista_e_task23_best","task23","vista_e_task23_best.pt",False,main=True)
    t23("vista_e_task23_max512","task23_max512","vista_e_task23_max512_best.pt",False)
    t23("vista_e_task23_max512_v2","task23_max512_v2","vista_e_task23_max512_v2_best.pt",False)
    t23("vista_e_task23_max512_R","task23_max512_R","vista_e_task23_max512_R_best.pt",False)
    t23("vista_e_task23_longformer","task23_longformer","vista_e_task23_longformer_best.pt",True)
    t23("vista_e_task23_longformer_v2","task23_longformer_v2","vista_e_task23_longformer_v2_best.pt",True)
    t23("vista_e_task23_longformer_R","task23_longformer_R","vista_e_task23_longformer_R_best.pt",True)
    print("\nBLOQUE 0 COMPLETO: 16 CSVs en", OUT)
