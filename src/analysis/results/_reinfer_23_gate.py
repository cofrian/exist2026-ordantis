"""Re-inferencia de los 7 checkpoints de Task 2.3 capturando la COMPUERTA DE SEXISMO REAL.
- 6 variantes (mod.infer -> ids, ps, pc, T, SX): ps = prob sexismo REAL (6a salida).
- principal task23 (infer -> ids, P, T, D): ps = max(prob categoria) (no tiene cabeza sexismo).
Guarda cache_23_gate.npz con, por checkpoint: ids, ps (sexismo pred), pc (5 cat pred), y el gold
comun T (soft cat 5) y SX (sexismo gold). Base para recomputar Bloques 2 y 5 (2.3)."""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config as C
if "task23" not in sys.modules:
    import task23
    if not hasattr(task23, "load_t23"): task23.load_t23 = task23.load_task23
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import importlib

XL = C.TEXT_MODEL; LF = "markussagen/xlm-roberta-longformer-base-4096"
ALT = os.path.join(C.OUT_DIR, "_alt")
VARIANTS = [
    ("vista_e_task23_max512",        "task23_max512",        "vista_e_task23_max512_best.pt",        XL),
    ("vista_e_task23_longformer",    "task23_longformer",    "vista_e_task23_longformer_best.pt",    LF),
    ("vista_e_task23_max512_v2",     "task23_max512_v2",     "vista_e_task23_max512_v2_best.pt",     XL),
    ("vista_e_task23_longformer_v2", "task23_longformer_v2", "vista_e_task23_longformer_v2_best.pt", LF),
    ("vista_e_task23_max512_R",      "task23_max512_R",      "vista_e_task23_max512_R_best.pt",      XL),
    ("vista_e_task23_longformer_R",  "task23_longformer_R",  "vista_e_task23_longformer_R_best.pt",  LF),
]

cache = {}
gold_T = gold_SX = gold_ids = None

def store(name, ids, ps, pc, T, SX):
    global gold_T, gold_SX, gold_ids
    cache[f"{name}__ids"] = np.array(ids)
    cache[f"{name}__ps"]  = np.asarray(ps, np.float32)
    cache[f"{name}__pc"]  = np.asarray(pc, np.float32)
    if gold_T is None:
        gold_ids = np.array(ids); gold_T = np.asarray(T, np.float32); gold_SX = np.asarray(SX, np.float32)

# --- 6 variantes: compuerta de sexismo REAL ---
for name, modname, ckfile, tk in VARIANTS:
    print(f"[infer] {name} (gate=cabeza sexismo real)", flush=True)
    mod = importlib.import_module(modname)
    tok = AutoTokenizer.from_pretrained(tk); cl = mod.collate(tok)
    splits = mod.load_t23()
    model = (getattr(mod, "VistaELong23", None) or getattr(mod, "VistaE23"))().to(C.DEVICE)
    sd = torch.load(os.path.join(ALT, ckfile), map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    dl = DataLoader(mod.DS23(splits["val"]), batch_size=16, shuffle=False, collate_fn=cl, num_workers=4)
    ids, ps, pc, T, SX = mod.infer(model, dl)
    del model; torch.cuda.empty_cache()
    store(name, ids, ps, pc, T, SX)

# --- principal: sin cabeza de sexismo -> gate = max(cat) ---
print("[infer] vista_e_task23_best (gate=max cat, sin cabeza sexismo)", flush=True)
tok = AutoTokenizer.from_pretrained(XL); cl = task23.collate(tok)
splits = task23.load_task23()
model = task23.VistaE23().to(C.DEVICE)
sd = torch.load(os.path.join(C.CKPT_DIR, "vista_e_task23_best.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
model.load_state_dict(sd, strict=False)
dl = DataLoader(task23.DS23(splits["val"]), batch_size=16, shuffle=False, collate_fn=cl, num_workers=4)
ids, P, T, D = task23.infer(model, dl)
del model; torch.cuda.empty_cache()
P = np.asarray(P, np.float32)
store("vista_e_task23_best", ids, P.max(1), P, T, gold_SX)  # gold_SX ya fijado por variantes
cache["gold__ids"] = gold_ids; cache["gold__T"] = gold_T; cache["gold__SX"] = gold_SX

np.savez(os.path.join(HERE, "cache_23_gate.npz"), **cache)
print("-> cache_23_gate.npz  (7 checkpoints + gold)")
