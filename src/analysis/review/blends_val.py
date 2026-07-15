"""Calcula en VALIDACION (n=598) las 3 celdas de blend que faltaban en la tabla por-run,
con las recetas confirmadas por el equipo:
  2.1 soft_1 = 0.6*P(vista_e_task21_max512_R) + 0.4*P(Gemini task2_1.sexist_probability)  [sobre P(YES)]
  2.2 soft_1 = blend 0.6/0.4 componente a componente sobre {NO,DIRECT,JUDG}, renormalizado
               modelo base = vista_e_task22_longformer ; P_Gemini = task2_2.intention_probabilities
  2.2 hard_3 = blend 0.6/0.4 SOLO en DIRECT y JUDG; P(NO)=1-(D+J); argmax
"""
import os, sys, json, tempfile, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
HERE=os.path.dirname(os.path.abspath(__file__))
import config as C, data as D
from dataset import MemeDataset, make_collate
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

G=json.load(open(os.path.join(C.PRE_DIR,"gemini_predictions.json")))
INT=["NO","DIRECT","JUDGEMENTAL"]; HIER={"YES":["DIRECT","JUDGEMENTAL"],"NO":[]}; TC="EXIST2025"

# ---------------- 2.1 soft_1 ----------------
def blend21():
    m=importlib.import_module("task21_max512_R")
    splits=D.load_split(); caps=m.load_caps()
    tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL); C.MAX_TOKENS=m.MAX_TOK
    vit={e["id"]:np.zeros(768,np.float32) for e in splits["val"]}
    dl=DataLoader(MemeDataset(splits["val"],tok,vit_emb=vit,captions=caps,use_caption=True),
                  batch_size=32,shuffle=False,collate_fn=make_collate(tok),num_workers=4)
    model=m.VistaE21().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt","vista_e_task21_max512_R_best.pt"),
                          map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,lo,tg=m.infer(model,dl); del model; torch.cuda.empty_cache()
    pE=1/(1+np.exp(-lo))
    pG=np.array([float((G.get(str(i),{}) or {}).get("task2_1",{}).get("sexist_probability",0.5) or 0.5) for i in ids])
    pB=0.6*pE+0.4*pG
    icms=m.icmsoft(ids,tg,pB)
    # F1+ e ICM a threshold optimo sobre ICM
    y=(tg>=0.5).astype(int); best=(0.5,-1e9)
    for t in [round(0.30+0.01*i,2) for i in range(41)]:
        ic=m.icm_hard(ids,y,(pB>=t).astype(int))
        if ic is not None and ic>best[1]: best=(t,ic)
    f1p=f1_score(y,(pB>=best[0]).astype(int))
    print(f"2.1 soft_1 blend  F1+={f1p:.4f}  ICM={best[1]:+.4f}(thr={best[0]})  ICMSoft={icms:+.4f}")
    return dict(F1=round(f1p,4),ICM=round(best[1],4),ICMSoft=round(icms,4))

# ---------------- 2.2 blends ----------------
def gem_int(mid):
    ip=((G.get(str(mid),{}) or {}).get("task2_2",{}) or {}).get("intention_probabilities",{}) or {}
    v=np.array([float(ip.get(k,0.0) or 0.0) for k in INT])
    s=v.sum(); return v/s if s>0 else np.array([1.0,0.0,0.0])

def pyevall22(ids,T,pred_or_probs,soft):
    with tempfile.TemporaryDirectory() as td:
        pf,gf=os.path.join(td,"p"),os.path.join(td,"g")
        gold=[{"test_case":TC,"id":str(m),"value":({INT[k]:float(T[i,k]) for k in range(3)} if soft else INT[int(np.argmax(T[i]))])} for i,m in enumerate(ids)]
        pr=[]
        for i,m in enumerate(ids):
            if soft:
                p=pred_or_probs[i]; p=p/max(p.sum(),1e-9); pr.append({"test_case":TC,"id":str(m),"value":{INT[k]:float(p[k]) for k in range(3)}})
            else: pr.append({"test_case":TC,"id":str(m),"value":INT[int(pred_or_probs[i])]})
        json.dump(pr,open(pf,"w")); json.dump(gold,open(gf,"w"))
        met=["ICMSoft","ICMSoftNorm"] if soft else ["ICM","ICMNorm"]
        rep=PyEvALLEvaluation().evaluate(pf,gf,met,**{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
            PyEvALLUtils.PARAM_HIERARCHY:HIER,PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        mm=rep.report["metrics"]; return {k:mm[k]["results"]["average_per_test_case"] for k in met if k in mm}

def blend22():
    m=importlib.import_module("task22_longformer")
    splits=m.load_t22(); tok=AutoTokenizer.from_pretrained(m.LONG_MODEL)
    dl=DataLoader(m.DS22(splits["val"]),batch_size=32,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
    model=m.VistaELong().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt","vista_e_task22_longformer_best.pt"),
                          map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,P,T=m.infer(model,dl); del model; torch.cuda.empty_cache()
    Pg=np.array([gem_int(i) for i in ids]); y=np.argmax(T,1)
    # soft_1: blend por clase, renormalizado
    Bs=0.6*P+0.4*Pg; Bs=Bs/Bs.sum(1,keepdims=True)
    f1s=f1_score(y,np.argmax(Bs,1),average="macro",labels=[0,1,2]); isf=pyevall22(ids,T,Bs,True)
    print(f"2.2 soft_1 blend  F1macro(argmax)={f1s:.4f}  ICMSoft={isf.get('ICMSoft',float('nan')):+.4f}")
    # hard_3: blend solo D/J, reconstruye NO, argmax
    Ph=np.zeros_like(P)
    Ph[:,1]=0.6*P[:,1]+0.4*Pg[:,1]; Ph[:,2]=0.6*P[:,2]+0.4*Pg[:,2]; Ph[:,0]=1-(Ph[:,1]+Ph[:,2])
    pred=np.argmax(Ph,1); f1h=f1_score(y,pred,average="macro",labels=[0,1,2]); ih=pyevall22(ids,T,pred,False)
    print(f"2.2 hard_3 blend  F1macro={f1h:.4f}  ICM={ih.get('ICM',float('nan')):+.4f}")
    return (dict(F1=round(f1s,4),ICMSoft=round(isf.get('ICMSoft',float('nan')),4)),
            dict(F1=round(f1h,4),ICM=round(ih.get('ICM',float('nan')),4)))

if __name__=="__main__":
    r21=blend21(); r22s,r22h=blend22()
    json.dump(dict(t21_soft1=r21,t22_soft1=r22s,t22_hard3=r22h), open(os.path.join(HERE,"_blends_val.json"),"w"),indent=2)
    print("guardado _blends_val.json")
