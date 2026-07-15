"""BLOQUE 4 — Parámetros y tamaño de los 16 checkpoints (sin inferencia)."""
import os, sys, csv, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import config as C
HERE=os.path.dirname(os.path.abspath(__file__))
CKPTS=[("M3_vista_E_best",C.CKPT_DIR),("vista_e_task22_best",C.CKPT_DIR),("vista_e_task23_best",C.CKPT_DIR)]
for n in ["vista_e_task21_max512","vista_e_task21_max512_R","vista_e_task21_longformer","vista_e_task21_longformer_R",
          "vista_e_task22_max512","vista_e_task22_max512_R","vista_e_task22_longformer",
          "vista_e_task23_max512","vista_e_task23_max512_v2","vista_e_task23_max512_R",
          "vista_e_task23_longformer","vista_e_task23_longformer_v2","vista_e_task23_longformer_R"]:
    CKPTS.append((n+"_best",os.path.join(C.OUT_DIR,"_alt")))
def is_buf(k): return k.endswith("position_ids") or k.endswith("token_type_ids")
rows=[]
for name,d in CKPTS:
    path=os.path.join(d,name+".pt")
    sd=torch.load(path,map_location="cpu",weights_only=False); sd=sd.get("model_state_dict",sd)
    tot=back=0
    for k,v in sd.items():
        if not torch.is_tensor(v) or is_buf(k): continue
        n_=v.numel(); tot+=n_
        if k.startswith("text_model."): back+=n_
    fam="Longformer" if "longformer" in name else "XLM-R base"
    rows.append(dict(checkpoint=name,familia=fam,total_params=tot,backbone_params=back,
                     head_params=tot-back,tamano_MB=round(os.path.getsize(path)/1e6,1)))
    print(f"  {name:34s} {fam:11s} total={tot:,} backbone={back:,} head={tot-back:,}")
with open(os.path.join(HERE,"parametros_todos.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("-> parametros_todos.csv")
