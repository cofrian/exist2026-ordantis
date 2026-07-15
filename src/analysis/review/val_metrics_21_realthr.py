"""Métricas de validación de los runs HARD de 2.1 a los THRESHOLDS REALES del v4
(recuperados por byte-match en test: hard_1=max512_R@0.46, hard_2=longformer@0.39,
hard_3=max512@0.51). ICMSoft de los soft (longformer, longformer_R) no depende de thr."""
import os, sys, importlib, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import config as C, data as D
from dataset import MemeDataset, make_collate

def val_probs(module_name, ckpt, kind):
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
    ids,lo,tg=m.infer(model,dl); del model; torch.cuda.empty_cache()
    return m, ids, 1/(1+np.exp(-lo)), tg

RUNS=[("hard_1","task21_max512_R","vista_e_task21_max512_R_best.pt","base",0.46),
      ("hard_2","task21_longformer","vista_e_task21_longformer_best.pt","long",0.39),
      ("hard_3","task21_max512","vista_e_task21_max512_best.pt","base",0.51)]
print("═══ 2.1 HARD runs — métricas de validación a los THRESHOLDS REALES del v4 ═══")
rows=[]
for run,mod,ck,kind,thr in RUNS:
    m,ids,p,tg=val_probs(mod,ck,kind); y=(tg>=0.5).astype(int); pred=(p>=thr).astype(int)
    f1=f1_score(y,pred); auc=roc_auc_score(y,p); icm=m.icm_hard(ids,y,pred); icms=m.icmsoft(ids,tg,p)
    print(f"  {run} ({mod.replace('task21_','')} @ thr={thr}):  F1+={f1:.4f}  AUC={auc:.4f}  ICM={icm:+.4f}  ICMSoft={icms:+.4f}")
    rows.append(dict(run=run,modelo=mod.replace("task21_",""),thr=thr,F1_pos=round(f1,4),AUC=round(auc,4),ICM=round(icm,4),ICMSoft=round(icms,4)))
import csv
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"metricas_2_1_thr_reales.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("  -> metricas_2_1_thr_reales.csv")
