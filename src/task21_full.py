"""Retraining Task 2.1 Vista E (M3) con train+val combinados (FINAL para entrega).

Pipeline:
  1) Cargar checkpoint actual (M3_vista_E_best.pt).
  2) Inferencia sobre val → buscar threshold óptimo en ICM hard.
  3) Combinar train+val, reentrenar Vista E (M3) desde warm-start del checkpoint.
     Fase 1 = 3 ép, Fase 2 = 8 ép sin early stopping (matching aproximado del run original).
  4) Inferencia sobre test, aplicar threshold del paso 2 + ensemble con Gemini.
  5) Generar 3 submissions hard/soft Task 2.1 (sobrescribe Ordantis_1/2/3).
"""
import json, os, csv, zipfile
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score

import config as C
import data as D
import evaluation_utils as E
from dataset import MemeDataset, make_collate, to_device
from models import MemeClassifier

SEED = 42
P1_EPOCHS = 3
P2_EPOCHS = 8
CKPT_ORIG = os.path.join(C.CKPT_DIR, "M3_vista_E_best.pt")
CKPT_FULL = os.path.join(C.CKPT_DIR, "M3_vista_E_full_best.pt")
THR_PATH = os.path.join(C.OUT_DIR, "_alt", "task21_threshold_val.json")
os.makedirs(os.path.dirname(THR_PATH), exist_ok=True)
DEV = C.DEVICE

CFG_E = dict(text=True, image=False, et=False, hr=False, eeg=True,
             caption=True, set_pool=True, emotions=True)


def load_captions():
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    caps, gprob = {}, {}
    for mid, v in g.items():
        if isinstance(v, dict):
            d = (v.get("description") or "").strip()
            a = (v.get("sexism_analysis") or "").strip()
            parts = []
            if d: parts.append("Description: " + d)
            if a: parts.append("Sexism Analysis: " + a)
            if parts: caps[str(mid)] = " ".join(parts)
            try: gprob[str(mid)] = float(v["task2_1"]["sexist_probability"])
            except Exception: pass
    return caps, gprob


@torch.no_grad()
def infer(model, dl):
    model.eval()
    ids, lo, tg = [], [], []
    for b in dl:
        b = to_device(b, DEV)
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            l = model(b)
        ids.extend(b["id"])
        lo.append(l.float().cpu().numpy())
        tg.append(b["soft"].cpu().numpy())
    return ids, np.concatenate(lo), np.concatenate(tg)


def search_threshold(ids, probs, tg):
    """Busca threshold que maximiza ICM hard en val."""
    best, bt = -1e9, 0.50
    for t in [round(0.30 + 0.01 * i, 2) for i in range(41)]:
        pred = probs >= t
        try:
            icm = E.eval_hard(ids, pred, tg).get("ICM", -1e9)
        except Exception:
            icm = -1e9
        if icm is not None and icm > best:
            best, bt = icm, t
    return bt


def run_epoch(model, dl_tr, opt, sched, pos_weight):
    model.train()
    for b in dl_tr:
        b = to_device(b, DEV)
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(b)
            tgt = b["soft"].clamp(0, 1)
            mask = (b["soft"] >= 0).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, tgt, weight=mask, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)


def main():
    C.set_seed(SEED); C.configure_gpu()
    print("[Task 2.1 Vista E FULL] retraining con train+val combinados", flush=True)
    caps, gprob = load_captions()
    splits = D.load_split()
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    collate = make_collate(tok)
    # Vit emb dummy (Vista E no usa imagen)
    dummy_vit = {e["id"]: np.zeros(768, dtype=np.float32) for e in splits["train"] + splits["val"] + splits["test"]}

    # ====== PASO 1: cargar modelo original y buscar threshold en val ======
    print("\n[PASO 1] Cargando M3_vista_E_best.pt y buscando threshold en val...", flush=True)
    model_orig = MemeClassifier(CFG_E).to(DEV)
    sd = torch.load(CKPT_ORIG, map_location="cpu", weights_only=False)["model_state_dict"]
    model_orig.load_state_dict(sd, strict=False)

    ds_va = MemeDataset(splits["val"], tok, dummy_vit, caps, True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, collate_fn=collate, num_workers=4)
    vids, vlo, vtg = infer(model_orig, dl_va)
    pE_v = 1 / (1 + np.exp(-vlo))
    thr_E = search_threshold(vids, pE_v, vtg)

    yv = (vtg >= 0.5).astype(int)
    pE_v_hard = (pE_v >= thr_E).astype(int)
    f1_orig = f1_score(yv, pE_v_hard)
    auc_orig = roc_auc_score(yv, pE_v)
    print(f"  [val] thr_E={thr_E:.2f}  F1+={f1_orig:.4f}  AUC={auc_orig:.4f}", flush=True)

    # Threshold ensemble (Vista E + Gemini)
    pG_v = np.array([gprob.get(str(i), 0.5) for i in vids])
    pEG_v = 0.6 * pE_v + 0.4 * pG_v
    thr_EG = search_threshold(vids, pEG_v, vtg)
    thr_G = search_threshold(vids, pG_v, vtg)
    print(f"  [val] thr_EG={thr_EG:.2f}  thr_G={thr_G:.2f}", flush=True)

    json.dump({"thr_E": thr_E, "thr_EG": thr_EG, "thr_G": thr_G}, open(THR_PATH, "w"))
    del model_orig; torch.cuda.empty_cache()

    # ====== PASO 2: reentrenar con train+val combinados ======
    print(f"\n[PASO 2] Reentrenando Vista E con train+val (n={len(splits['train']) + len(splits['val'])})...", flush=True)
    full_train = splits["train"] + splits["val"]
    ds_tr = MemeDataset(full_train, tok, dummy_vit, caps, True)
    dl_tr = DataLoader(ds_tr, batch_size=24, shuffle=True, collate_fn=collate,
                       num_workers=4, pin_memory=True)

    model = MemeClassifier(CFG_E).to(DEV)
    # Warm-start desde el checkpoint original
    model.load_state_dict(sd, strict=False)
    pos_weight = torch.tensor([C.POS_WEIGHT], device=DEV)

    def quick_log(tag):
        ids, lo, tg = infer(model, dl_va)
        p = 1 / (1 + np.exp(-lo)); pred = (p >= thr_E).astype(int)
        y = (tg >= 0.5).astype(int)
        print(f"  [{tag}] F1+(val-in-train)={f1_score(y, pred):.4f}  AUC={roc_auc_score(y, p):.4f}", flush=True)

    # Fase 1: backbone congelado
    from models import _layer_index
    for p in model.text_model.parameters(): p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  FASE 1 ({P1_EPOCHS} ép, head warm-up, {sum(p.numel() for p in trainable):,} params)", flush=True)
    opt = torch.optim.AdamW(trainable, lr=C.PHASE1_LR, weight_decay=C.WEIGHT_DECAY)
    st = max(1, len(dl_tr) * P1_EPOCHS)
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP_FRAC * st), st)
    for ep in range(1, P1_EPOCHS + 1):
        run_epoch(model, dl_tr, opt, sched, pos_weight); quick_log(f"F1 {ep}/{P1_EPOCHS}")
    del opt, sched; torch.cuda.empty_cache()

    # Fase 2: full fine-tune, sin early stopping
    for p in model.text_model.parameters(): p.requires_grad = True
    low, high = [], []
    for n_, p in model.text_model.named_parameters():
        if "embeddings" in n_:
            low.append(p)
        elif "encoder.layer." in n_:
            ln = int(n_.split("encoder.layer.")[1].split(".")[0])
            (low if ln <= 6 else high).append(p)
        else:
            high.append(p)
    head = [p for n_, p in model.named_parameters() if not n_.startswith("text_model.")]
    opt = torch.optim.AdamW([
        {"params": low, "lr": C.LR_LOW},
        {"params": high, "lr": C.LR_HIGH},
        {"params": head, "lr": C.LR_HEAD},
    ], weight_decay=C.WEIGHT_DECAY)
    st = max(1, len(dl_tr) * P2_EPOCHS)
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP_FRAC * st), st)
    print(f"  FASE 2 ({P2_EPOCHS} ép, sin early stopping)", flush=True)
    for ep in range(1, P2_EPOCHS + 1):
        run_epoch(model, dl_tr, opt, sched, pos_weight); quick_log(f"F2 {ep}/{P2_EPOCHS}")

    sd_final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(model_state_dict=sd_final), CKPT_FULL)
    print(f"  checkpoint final -> {CKPT_FULL}", flush=True)

    # ====== PASO 3: inferencia test ======
    print(f"\n[PASO 3] Inferencia sobre test, aplicando threshold del PASO 1...", flush=True)
    ds_te = MemeDataset(splits["test"], tok, dummy_vit, caps, True)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, collate_fn=collate, num_workers=4)
    tids, tlo, _ = infer(model, dl_te)
    pE_t = 1 / (1 + np.exp(-tlo))
    pG_t = np.array([gprob.get(str(i), 0.5) for i in tids])
    pEG_t = 0.6 * pE_t + 0.4 * pG_t

    print(f"  test n={len(tids)}, sin Gemini: {sum(1 for i in tids if str(i) not in gprob)}", flush=True)

    # ====== PASO 4: submissions ======
    OUT = C.OUT_DIR
    runs = {
        ("1", "soft"): pE_t,
        ("1", "hard"): (pE_t >= thr_E),
        ("2", "soft"): pEG_t,
        ("2", "hard"): (pEG_t >= thr_EG),
        ("3", "soft"): pG_t,
        ("3", "hard"): (pG_t >= thr_G),
    }
    for (n, kind), arr in runs.items():
        path = os.path.join(OUT, f"task2_1_{kind}_{C.TEAM_NAME}_{n}")
        if kind == "hard":
            D.write_hard_file(path, tids, np.asarray(arr).astype(bool))
        else:
            D.write_soft_file(path, tids, np.asarray(arr, dtype=float))
        print(f"  escrito task2_1_{kind}_{C.TEAM_NAME}_{n}", flush=True)

    # ====== PASO 5: rezip ======
    print("\n[PASO 5] Regenerando zip final...", flush=True)
    fns = sorted(f for f in os.listdir(OUT) if f.startswith("task2_"))
    zp = os.path.join(C.ROOT, f"exist2026_{C.TEAM_NAME}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in fns:
            zf.write(os.path.join(OUT, fn), arcname=os.path.join(f"exist2026_{C.TEAM_NAME}", fn))
    print(f"  ZIP: {zp}  ({os.path.getsize(zp)/1024:.1f} KB, {len(fns)} ficheros)", flush=True)
    print("\n=== TERMINADO Task 2.1 Vista E FULL ===", flush=True)


if __name__ == "__main__":
    main()
