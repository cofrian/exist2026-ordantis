"""(A) VALIDA la fidelidad del pipeline de inferencia contra un fichero v4 que sobrevive
    (task2_3_soft_Ordantis_2 = XLM-R 512 sin sampler, prob directa): regenera esa salida
    y reporta % de coincidencia. Prueba limpia (sin thresholds).
(B) REGENERA el v4 de Task 2.1 en test -> exist2026_Ordantis/_v4_regenerado/ (NO pisa la
    carpeta principal). soft_1/2/3 son deterministas (bit-fieles); hard_1/2/3 usan los
    thresholds optimizados en validacion (0.61/0.57/0.60) porque los del v4 no se guardaron.
"""
import os, sys, json, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
HERE=os.path.dirname(os.path.abspath(__file__))
import config as C, data as D
from dataset import MemeDataset, make_collate

OUT=C.OUT_DIR
G=json.load(open(os.path.join(C.PRE_DIR,"gemini_predictions.json")))

def probs_task21(module_name, ckpt, kind):
    """Devuelve dict id->prob_sigmoide en TEST para un checkpoint 2.1."""
    m=importlib.import_module(module_name)
    splits=D.load_split(); caps=m.load_caps()
    if kind=="base":
        tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL); C.MAX_TOKENS=m.MAX_TOK
        vit={e["id"]:np.zeros(768,np.float32) for e in splits["test"]}
        dl=DataLoader(MemeDataset(splits["test"],tok,vit_emb=vit,captions=caps,use_caption=True),
                      batch_size=32,shuffle=False,collate_fn=make_collate(tok),num_workers=4)
        model=m.VistaE21().to(C.DEVICE)
    else:
        tok=AutoTokenizer.from_pretrained(m.LONG_MODEL)
        dl=DataLoader(m.DS21(splits["test"],caps),batch_size=8,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
        model=m.VistaELong21().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt",ckpt),map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,lo,_=m.infer(model,dl); del model; torch.cuda.empty_cache()
    p=1/(1+np.exp(-lo))
    return {str(i):float(pp) for i,pp in zip(ids,p)}

# ===================== (A) FIDELIDAD contra v4 2.3 soft_2 =====================
def validate_fidelity():
    print("=== (A) Validación de fidelidad: regenerar 2.3 soft_2 y comparar con el v4 superviviente ===", flush=True)
    orig_path=os.path.join(OUT,"task2_3_soft_Ordantis_2")
    if not os.path.exists(orig_path):
        print("  [no existe el v4 superviviente, salto]"); return
    orig={d["id"]:d["value"] for d in json.load(open(orig_path))}
    m=importlib.import_module("task23_max512")
    splits=m.load_t23(); tok=AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    dl=DataLoader(m.DS23(splits["test"]),batch_size=16,shuffle=False,collate_fn=m.collate(tok),num_workers=4)
    ModelCls=getattr(m,"VistaELong23",None) or getattr(m,"VistaE23")
    model=ModelCls().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.OUT_DIR,"_alt","vista_e_task23_max512_best.pt"),map_location="cpu",weights_only=False)["model_state_dict"],strict=False)
    ids,ps,pc,T,SX=m.infer(model,dl); del model; torch.cuda.empty_cache()
    CATS=["IDEOLOGICAL-INEQUALITY","STEREOTYPING-DOMINANCE","OBJECTIFICATION","SEXUAL-VIOLENCE","MISOGYNY-NON-SEXUAL-VIOLENCE"]
    maxdiff=0.0; within3=0; within2=0; n=0
    for k,i in enumerate(ids):
        i=str(i)
        if i not in orig: continue
        regen={c:float(pc[k,j]) for j,c in enumerate(CATS)}; regen["NO"]=float(max(0.0,1.0-max(pc[k])))
        for key in regen:
            d=abs(regen[key]-float(orig[i].get(key,0.0))); maxdiff=max(maxdiff,d)
            within3+= d<1e-3; within2+= d<1e-2; n+=1
    print(f"  comparados {n} valores (memes×clases). max|Δ|={maxdiff:.5f}")
    print(f"  coincidencia <1e-3: {100*within3/max(n,1):.2f}%   <1e-2: {100*within2/max(n,1):.2f}%")
    if maxdiff<1e-2: print("  -> pipeline de inferencia FIEL (reproduce el v4 superviviente).")
    else: print("  -> DIVERGENCIA: el v4 2.3 soft_2 se generó con otra config/checkpoint que la asumida.")

# ===================== (B) REGENERAR v4 de 2.1 =====================
def regenerate_task21():
    print("\n=== (B) Regeneración del v4 de Task 2.1 (staging: _v4_regenerado/) ===", flush=True)
    dst=os.path.join(OUT,"_v4_regenerado"); os.makedirs(dst,exist_ok=True)
    # ids de test en orden canónico
    test_ids=[e["id"] for e in D.load_split()["test"]]
    pG={i:float((G.get(str(i),{}) or {}).get("task2_1",{}).get("sexist_probability",0.5) or 0.5) for i in test_ids}
    print("  infiriendo max512_R ...", flush=True); p_m512R=probs_task21("task21_max512_R","vista_e_task21_max512_R_best.pt","base")
    print("  infiriendo longformer ...", flush=True); p_lf=probs_task21("task21_longformer","vista_e_task21_longformer_best.pt","long")
    print("  infiriendo max512 ...", flush=True); p_m512=probs_task21("task21_max512","vista_e_task21_max512_best.pt","base")
    print("  infiriendo longformer_R ...", flush=True); p_lfR=probs_task21("task21_longformer_R","vista_e_task21_longformer_R_best.pt","long")
    THR={"hard_1":0.61,"hard_2":0.57,"hard_3":0.60}   # optimizados en validación (v4 no guardó los suyos)
    def arr(d): return np.array([d[str(i)] for i in test_ids])
    pm512R=arr(p_m512R); plf=arr(p_lf); pm512=arr(p_m512); plfR=arr(p_lfR); pg=np.array([pG[i] for i in test_ids])
    soft1=0.6*pm512R+0.4*pg
    files={
        "task2_1_hard_Ordantis_1":("hard",(pm512R>=THR["hard_1"])),
        "task2_1_hard_Ordantis_2":("hard",(plf>=THR["hard_2"])),
        "task2_1_hard_Ordantis_3":("hard",(pm512>=THR["hard_3"])),
        "task2_1_soft_Ordantis_1":("soft",soft1),
        "task2_1_soft_Ordantis_2":("soft",plf),
        "task2_1_soft_Ordantis_3":("soft",plfR),
    }
    for fn,(kind,a) in files.items():
        p=os.path.join(dst,fn)
        if kind=="hard": D.write_hard_file(p,test_ids,np.asarray(a).astype(bool))
        else: D.write_soft_file(p,test_ids,np.asarray(a,dtype=float))
        yes=int(np.sum(np.asarray(a)>=0.5)) if kind=="hard" else int(np.sum(np.asarray(a)>=0.5))
        print(f"    escrito {fn}  ({'YES='+str(yes) if kind=='hard' else 'soft'})")
    json.dump({"thresholds_usados":THR,"nota":"soft deterministas; hard con thr optimizados en val (v4 no guardó los suyos)"},
              open(os.path.join(dst,"_REGEN_INFO.json"),"w"),indent=2)
    print("  -> 6 ficheros en", dst)

if __name__=="__main__":
    validate_fidelity()
    regenerate_task21()
