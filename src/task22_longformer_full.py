"""Retraining Task 2.2 Longformer con train+val combinados (FINAL para entrega).

Pipeline:
  1) Cargar checkpoint actual (vista_e_task22_longformer_best.pt).
  2) Inferencia sobre val → buscar thresholds (t_J, t_D) y Platt.
  3) Combinar train+val, reentrenar Longformer (P1=2, P2=8 fijo, sin early stopping).
  4) Inferencia sobre test, aplicar thresholds y Platt del paso 2.
  5) Generar 3 submissions hard/soft Task 2.2 (sobrescribe Ordantis_1/2/3).
"""
import json, os, tempfile, math, zipfile
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import config as C
import data as D
from models import SetAttentionPool

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
INT = ["NO", "DIRECT", "JUDGEMENTAL"]
HIER = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
TC = "EXIST2025"; EPS = 1e-7; SEED = 999
LONG_MODEL = "markussagen/xlm-roberta-longformer-base-4096"
MAX_TOK = 1100
P1, P2 = 2, 8
DEV = C.DEVICE
CKPT_ORIG = os.path.join(ALT, "vista_e_task22_longformer_best.pt")
CKPT_FULL = os.path.join(ALT, "vista_e_task22_longformer_full_best.pt")

# Reutilizo carga y dataset del task22_longformer
from task22_longformer import load_t22, DS22, HierHead, VistaELong, sampler_for, loss_fn


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j, :n] = x["eeg"]; mask[j, :n] = True
        gam = torch.zeros_like(enc["input_ids"]); gam[:, 0] = 1
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    global_attention_mask=gam,
                    feat=torch.stack([x["feat"] for x in b]), emotions=torch.stack([x["emotions"] for x in b]),
                    eeg=eeg, eeg_mask=mask, soft=torch.stack([x["soft"] for x in b]))
    return f


@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, P, T = [], [], []
    for b in dl:
        bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            probs, _, _ = model(bb)
        ids += b["id"]; P.append(probs.float().cpu().numpy()); T.append(b["soft"].numpy())
    return ids, np.concatenate(P), np.concatenate(T)


def dec(p, tj, td):
    return np.where(p[:, 2] > tj, 2, np.where(p[:, 1] > td, 1, 0))


def search_thresholds(model, dl_va):
    """Busca (tj, td) en val maximizando 0.5*F1macro + 0.5*ICMNorm."""
    from pyevall.evaluation import PyEvALLEvaluation
    from pyevall.utils.utils import PyEvALLUtils
    ids, P, T = infer(model, dl_va)
    y = np.argmax(T, 1)

    def f1m(pred):
        return f1_score(y, pred, average="macro", labels=[0, 1, 2])

    def icm_hard(pred):
        with tempfile.TemporaryDirectory() as td:
            pf, gf = os.path.join(td, "p"), os.path.join(td, "g")
            json.dump([{"test_case": TC, "id": str(i), "value": INT[int(pred[k])]} for k, i in enumerate(ids)], open(pf, "w"))
            json.dump([{"test_case": TC, "id": str(i), "value": INT[int(y[k])]} for k, i in enumerate(ids)], open(gf, "w"))
            rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICMNorm"],
                **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
                   PyEvALLUtils.PARAM_HIERARCHY: HIER,
                   PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
            return rep.report["metrics"]["ICMNorm"]["results"]["average_per_test_case"]

    best, bestT = -1e9, (0.30, 0.40)
    for tj in np.arange(0.10, 0.45, 0.025):
        for td in np.arange(0.25, 0.60, 0.025):
            pred = dec(P, tj, td)
            f1 = f1m(pred)
            try: icmn = icm_hard(pred)
            except Exception: icmn = -1.0
            sc = 0.5 * f1 + 0.5 * icmn
            if sc > best: best, bestT = sc, (float(tj), float(td))
    tj, td = bestT
    pred = dec(P, tj, td)
    f1 = f1m(pred)
    print(f"  [val] t_JUDG={tj:.3f} t_DIRECT={td:.3f}  F1macro={f1:.4f}", flush=True)

    # Platt scaling 3 clases
    oh = np.eye(3)[y]
    platt = []
    for c in range(3):
        lg = np.log(np.clip(P[:, c], EPS, 1 - EPS) / np.clip(1 - P[:, c], EPS, 1 - EPS))
        lr = LogisticRegression().fit(lg.reshape(-1, 1), oh[:, c])
        platt.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
    return tj, td, platt


def apply_platt(P, platt):
    Pc = np.zeros_like(P)
    for c, (a, b) in enumerate(platt):
        lg = np.log(np.clip(P[:, c], EPS, 1 - EPS) / np.clip(1 - P[:, c], EPS, 1 - EPS))
        Pc[:, c] = 1 / (1 + np.exp(-(a * lg + b)))
    s = Pc.sum(1, keepdims=True).clip(min=EPS)
    return Pc / s


_G = None
def _gemini_intprobs(mid):
    global _G
    if _G is None:
        _G = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    v = _G.get(str(mid))
    if isinstance(v, dict):
        ip = (v.get("task2_2", {}) or {}).get("intention_probabilities", {}) or {}
        p = np.array([float(ip.get(k, 0.0) or 0.0) for k in INT], np.float32)
        s = p.sum()
        return p / s if s > 0 else np.array([1, 0, 0], np.float32)
    return np.array([1, 0, 0], np.float32)


def main():
    C.set_seed(SEED); C.configure_gpu()
    print(f"[Task 2.2 Longformer FULL] {LONG_MODEL} MAX_TOK={MAX_TOK}", flush=True)
    splits = load_t22()
    tok = AutoTokenizer.from_pretrained(LONG_MODEL); cl = collate(tok)

    # ====== PASO 1: cargar modelo original y buscar thresholds en val ======
    print("\n[PASO 1] Cargando checkpoint Longformer original y buscando thresholds en val...", flush=True)
    dl_va_only = DataLoader(DS22(splits["val"]), batch_size=8, shuffle=False, collate_fn=cl, num_workers=4)
    model_orig = VistaELong().to(DEV)
    if os.path.exists(CKPT_ORIG):
        sd = torch.load(CKPT_ORIG, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        model_orig.load_state_dict(sd)
        print(f"  modelo original cargado de {CKPT_ORIG}", flush=True)
        tj_orig, td_orig, platt_orig = search_thresholds(model_orig, dl_va_only)
        json.dump({"tj": tj_orig, "td": td_orig, "platt": platt_orig},
                  open(os.path.join(ALT, "task22_longformer_thresholds_val.json"), "w"))
    else:
        print(f"  ⚠️ no existe {CKPT_ORIG} — usaré defaults", flush=True)
        tj_orig, td_orig, platt_orig = 0.30, 0.40, [(1.0, 0.0)] * 3

    del model_orig; torch.cuda.empty_cache()

    # ====== PASO 2: reentrenar con train+val ======
    print(f"\n[PASO 2] Reentrenando con train+val (n={len(splits['train']) + len(splits['val'])})...", flush=True)
    full_train = splits["train"] + splits["val"]
    dl_tr = DataLoader(DS22(full_train), batch_size=4, sampler=sampler_for(full_train),
                       collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS22(splits["val"]), batch_size=8, shuffle=False, collate_fn=cl, num_workers=4)

    model = VistaELong().to(DEV)

    def run_epoch(opt, sch):
        model.train()
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                probs, lb, lt = model(bb); loss = loss_fn(probs, lb, lt, bb["soft"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

    def quick_log(tag):
        ids, P, T = infer(model, dl_va); y = np.argmax(T, 1)
        f1 = f1_score(y, np.argmax(P, 1), average="macro", labels=[0, 1, 2])
        print(f"  [{tag}] F1macro(val-in-train)={f1:.4f}", flush=True)

    # Fase 1
    for p in model.text_model.parameters(): p.requires_grad = False
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"  FASE 1 ({P1} ép, {sum(p.numel() for p in tr):,} params)", flush=True)
    opt = torch.optim.AdamW(tr, lr=5e-5, weight_decay=0.01)
    st = max(1, len(dl_tr) * P1); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    for e in range(1, P1 + 1): run_epoch(opt, sch); quick_log(f"F1 {e}/{P1}")
    del opt, sch; torch.cuda.empty_cache()

    # Fase 2
    for p in model.text_model.parameters(): p.requires_grad = True
    low, high = [], []
    for n, p in model.text_model.named_parameters():
        ln = int(n.split("encoder.layer.")[1].split(".")[0]) if "encoder.layer." in n else None
        (low if (("embeddings" in n) or (ln is not None and ln <= 6)) else high).append(p)
    headp = [p for n, p in model.named_parameters() if not n.startswith("text_model.")]
    opt = torch.optim.AdamW([{"params": low, "lr": 1e-5}, {"params": high, "lr": 3e-5},
                             {"params": headp, "lr": 1e-4}], weight_decay=0.01)
    st = max(1, len(dl_tr) * P2); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    print(f"  FASE 2 ({P2} ép, sin early stopping)", flush=True)
    for e in range(1, P2 + 1):
        run_epoch(opt, sch); quick_log(f"F2 {e}/{P2}")

    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(model_state_dict=sd), CKPT_FULL)
    print(f"  checkpoint final -> {CKPT_FULL}", flush=True)

    # ====== PASO 3: inferencia test ======
    print(f"\n[PASO 3] Inferencia sobre test, aplicando thresholds y Platt del PASO 1...", flush=True)
    dl_te = DataLoader(DS22(splits["test"]), batch_size=4, shuffle=False, collate_fn=cl, num_workers=4)
    tids, tP, _ = infer(model, dl_te)
    tPc = apply_platt(tP, platt_orig)
    g_test = np.array([_gemini_intprobs(i) for i in tids])
    blend = 0.6 * tP + 0.4 * g_test

    # ====== PASO 4: submissions ======
    OUT = C.OUT_DIR
    runs = {
        ("1", "soft"): (g_test, None),
        ("1", "hard"): (g_test, np.argmax(g_test, 1)),
        ("2", "soft"): (blend, None),
        ("2", "hard"): (blend, np.argmax(blend, 1)),
        ("3", "soft"): (tPc, None),                          # Vista E con Platt
        ("3", "hard"): (tP, dec(tP, tj_orig, td_orig)),      # Vista E con thresholds 2D
    }
    for (n, kind), (probs, hard) in runs.items():
        out = []
        for i, mid in enumerate(tids):
            if kind == "hard":
                out.append({"test_case": TC, "id": str(mid), "value": INT[int(hard[i])]})
            else:
                p = probs[i]; p = p / max(p.sum(), EPS)
                out.append({"test_case": TC, "id": str(mid),
                            "value": {"NO": float(p[0]), "DIRECT": float(p[1]), "JUDGEMENTAL": float(p[2])}})
        json.dump(out, open(os.path.join(OUT, f"task2_2_{kind}_{C.TEAM_NAME}_{n}"), "w"), indent=2)
        print(f"  escrito task2_2_{kind}_{C.TEAM_NAME}_{n}", flush=True)

    # ====== PASO 5: rezip ======
    print("\n[PASO 5] Regenerando zip final...", flush=True)
    fns = sorted(f for f in os.listdir(OUT) if f.startswith("task2_"))
    zp = os.path.join(C.ROOT, f"exist2026_{C.TEAM_NAME}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in fns:
            zf.write(os.path.join(OUT, fn), arcname=os.path.join(f"exist2026_{C.TEAM_NAME}", fn))
    print(f"  ZIP: {zp}  ({os.path.getsize(zp)/1024:.1f} KB, {len(fns)} ficheros)", flush=True)
    print("\n=== TERMINADO Task 2.2 Longformer FULL ===", flush=True)


if __name__ == "__main__":
    main()
