"""TAREA 4 - Ablacion del aporte de Gemini (validacion, mismo checkpoint, sin reentrenar).

2.2 (VistaE22): pone a CEROS en el forward las 7 features numericas de Gemini (gfeat)
    y compara ICM / F1macro con la version normal. (ablacion pedida, exacta.)
2.3 (VistaE23): NO tiene features numericas de Gemini en el forward (solo texto 768 + EEG 256
    + Ekman 7). Poner '6 features a cero' es N/A. En su lugar se hace la ablacion EQUIVALENTE:
    se elimina el TEXTO derivado de Gemini (description/analysis/reasoning/category_reasoning),
    dejando solo el OCR, y se compara. Asi se cuantifica el aporte real de Gemini en 2.3.
"""
import os, sys, json, tempfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
import config as C
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

rows = []

# ================= 2.2 ablacion numerica =================
def ablate_22():
    import task22 as T22
    INT = ["NO","DIRECT","JUDGEMENTAL"]; HIER = {"YES":["DIRECT","JUDGEMENTAL"],"NO":[]}; TC="EXIST2025"
    splits = T22.load_task22()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    dl = DataLoader(T22.DS22(splits["val"], tok), batch_size=64, shuffle=False, collate_fn=T22.collate(tok), num_workers=4)
    model = T22.VistaE22().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR,"vista_e_task22_best.pt"),
                          map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
    model.eval()

    @torch.no_grad()
    def infer(zero_gfeat):
        ids, P, T = [], [], []
        for b in dl:
            bb = {k:(v.to(C.DEVICE) if torch.is_tensor(v) else v) for k,v in b.items()}
            if zero_gfeat: bb["gfeat"] = torch.zeros_like(bb["gfeat"])
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                probs,_,_ = model(bb)
            ids += b["id"]; P.append(probs.float().cpu().numpy()); T.append(b["soft"].numpy())
        return ids, np.concatenate(P), np.concatenate(T)

    def pyevall_icm(ids, y, pred_idx_or_probs, soft):
        with tempfile.TemporaryDirectory() as td:
            pf, gf = os.path.join(td,"p"), os.path.join(td,"g")
            gold, preds = [], []
            for i,mid in enumerate(ids):
                if soft:
                    gold.append({"test_case":TC,"id":str(mid),"value":{INT[k]:float(y[i,k]) for k in range(3)}})
                    p = pred_idx_or_probs[i]; p=p/max(p.sum(),1e-9)
                    preds.append({"test_case":TC,"id":str(mid),"value":{INT[k]:float(p[k]) for k in range(3)}})
                else:
                    gold.append({"test_case":TC,"id":str(mid),"value":INT[int(np.argmax(y[i]))]})
                    preds.append({"test_case":TC,"id":str(mid),"value":INT[int(pred_idx_or_probs[i])]})
            json.dump(preds,open(pf,"w")); json.dump(gold,open(gf,"w"))
            metrics = ["ICMSoft","ICMSoftNorm"] if soft else ["ICM","ICMNorm"]
            rep = PyEvALLEvaluation().evaluate(pf,gf,metrics,
                **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
                   PyEvALLUtils.PARAM_HIERARCHY:HIER, PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
            m = rep.report["metrics"]
            return {k:m[k]["results"]["average_per_test_case"] for k in metrics if k in m}

    def thr2d(P,T):
        y=np.argmax(T,1); best=(-1,0.30,0.40)
        for tj in np.arange(0.10,0.45,0.025):
            for td in np.arange(0.25,0.60,0.025):
                pred=np.where(P[:,2]>tj,2,np.where(P[:,1]>td,1,0))
                s=f1_score(y,pred,average="macro",labels=[0,1,2])
                if s>best[0]: best=(s,tj,td)
        return best[1],best[2]

    for tag, z in [("2.2 normal (con gfeat Gemini)", False), ("2.2 ablacion (gfeat=0)", True)]:
        ids,P,T = infer(z); y=np.argmax(T,1)
        tj,td = thr2d(P,T); pred=np.where(P[:,2]>tj,2,np.where(P[:,1]>td,1,0))
        f1m=f1_score(y,pred,average="macro",labels=[0,1,2])
        f1c=f1_score(y,pred,average=None,labels=[0,1,2])
        ih=pyevall_icm(ids,T,pred,False); isf=pyevall_icm(ids,T,P,True)
        print(f"  {tag:32s} F1macro={f1m:.4f} F1[N/D/J]={f1c[0]:.2f}/{f1c[1]:.2f}/{f1c[2]:.2f} "
              f"ICM={ih.get('ICM',float('nan')):+.4f} ICMSoft={isf.get('ICMSoft',float('nan')):+.4f}", flush=True)
        rows.append(dict(subtask="2.2", condicion=tag, ablacion=("gfeat=0" if z else "baseline"),
                         F1macro=round(f1m,4), F1_NO=round(f1c[0],4), F1_DIRECT=round(f1c[1],4), F1_JUDG=round(f1c[2],4),
                         ICM=round(ih.get("ICM",float("nan")),4), ICMNorm=round(ih.get("ICMNorm",float("nan")),4),
                         ICMSoft=round(isf.get("ICMSoft",float("nan")),4)))
    del model; torch.cuda.empty_cache()

# ================= 2.3 ablacion de texto Gemini =================
def ablate_23():
    import task23 as T23
    import eval_main23 as EM  # tiene guard __main__, importa helpers sin ejecutar
    splits = T23.load_task23()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL); cl = T23.collate(tok)
    model = T23.VistaE23().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR,"vista_e_task23_best.pt"),
                          map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
    model.eval()
    # gold sexista real
    raw = json.load(open(C.TRAIN_JSON, encoding="utf-8")); sex_real={}
    for m in raw.values():
        t1=[v for v in m.get("labels_task2_1",[]) if v in ("YES","NO")]; n=len(t1) or 1
        sex_real[str(m["id_EXIST"])]=sum(1 for v in t1 if v=="YES")/n

    def run(strip_gemini):
        val = [dict(e) for e in splits["val"]]
        if strip_gemini:
            for e in val: e["text23"] = e["text"]   # solo OCR, sin texto de Gemini
        dl = DataLoader(T23.DS23(val), batch_size=16, shuffle=False, collate_fn=cl, num_workers=4)
        ids,P,T,_ = T23.infer(model, dl)
        SX = np.array([sex_real[i] for i in ids]); ps = P.max(1)
        tsex,tcat = EM.find_best_thr(ids, ps, P, T, SX)
        gh = EM.gold_hard_from_soft(T, SX); pr = EM.pred_from_probs(ps, P, tsex, tcat)
        icm,icmn,fm = EM.pyevall_hard_full(ids, gh, pr)
        icms,icmsn = EM.pyevall_soft_full(ids, T, SX, ps, P)
        return fm, icm, icmn, icms, icmsn, tsex, tcat[0]

    for tag, strip in [("2.3 normal (texto con Gemini)", False), ("2.3 ablacion (solo OCR, sin texto Gemini)", True)]:
        fm,icm,icmn,icms,icmsn,tsex,tc = run(strip)
        print(f"  {tag:44s} F1macro={fm:.4f} ICM={icm:+.4f} ICMNorm={icmn:.4f} ICMSoft={icms:+.4f}", flush=True)
        rows.append(dict(subtask="2.3", condicion=tag, ablacion=("texto_gemini_off" if strip else "baseline"),
                         F1macro=round(fm,4), F1_NO="", F1_DIRECT="", F1_JUDG="",
                         ICM=round(icm,4), ICMNorm=round(icmn,4), ICMSoft=round(icms,4)))
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    import csv
    print("### 2.2 ablacion numerica (gfeat=0) ###")
    ablate_22()
    print("### 2.3 ablacion de texto Gemini (N/A features numericas) ###")
    ablate_23()
    cols = ["subtask","condicion","ablacion","F1macro","F1_NO","F1_DIRECT","F1_JUDG","ICM","ICMNorm","ICMSoft"]
    out = os.path.join(HERE, "ablacion_gemini_features.csv")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("CSV:", out)
