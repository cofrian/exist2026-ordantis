"""TAREA 5 - Numero de parametros por checkpoint (backbone incluido).
Cuenta desde el model_state_dict guardado (params reales entrenados). Desglosa
backbone de texto (text_model.*) vs resto (cabezas + EEG pool + aux + Ekman/Gemini).
Excluye el buffer no-parametro 'position_ids' de XLM-R para igualar
sum(p.numel() for p in model.parameters())."""
import os, sys, csv
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
import config as C

CKPTS = []
for d, tag in [(C.CKPT_DIR, "checkpoints"), (os.path.join(C.OUT_DIR, "_alt"), "_alt")]:
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".pt"): CKPTS.append((os.path.join(d, fn), fn, tag))

def is_buffer(k):
    return k.endswith("position_ids") or k.endswith("token_type_ids")

rows = []
for path, fn, tag in CKPTS:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    total = back = rest = 0
    for k, v in sd.items():
        if not torch.is_tensor(v) or is_buffer(k): continue
        n = v.numel(); total += n
        if k.startswith("text_model."): back += n
        else: rest += n
    size_mb = os.path.getsize(path) / 1e6
    rows.append(dict(checkpoint=fn, dir=tag, params_total=total,
                     params_backbone_texto=back, params_resto=rest,
                     M_params=round(total/1e6, 3), file_MB=round(size_mb, 1)))
    print(f"  {fn:42s} total={total:>12,}  ({total/1e6:6.2f}M)  backbone={back/1e6:5.1f}M  resto={rest/1e6:5.2f}M")

with open(os.path.join(HERE, "parametros.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} checkpoints -> parametros.csv")
print("Rango:", min(r["params_total"] for r in rows), "-", max(r["params_total"] for r in rows), "params")
