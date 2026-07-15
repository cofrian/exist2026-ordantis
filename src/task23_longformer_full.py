"""Retraining Task 2.3 Longformer con train+val combinados (FINAL para entrega).

Pipeline:
  1) Cargar checkpoint actual (vista_e_task23_longformer_best.pt).
  2) Inferencia sobre val con ese modelo → buscar thresholds (tsex, tcat) honestos.
  3) Combinar train+val, reentrenar Longformer desde cero (warm-start desde task22)
     con número FIJO de épocas (matching de steps), sin early stopping.
  4) Inferencia sobre test con el modelo retrained.
  5) Aplicar los thresholds del paso 2.
  6) Generar submissions hard/soft de Task 2.3 (sobrescribe Ordantis_1/2/3).

Genera también el zip final.
"""
import json, os, tempfile, math
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import config as C
import data as D
from models import SetAttentionPool

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
LONG_MODEL = "markussagen/xlm-roberta-longformer-base-4096"
CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
        "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]
ALL = ["NO"] + CATS; HIER = {"YES": CATS, "NO": []}
TC = "EXIST2025"; EPS = 1e-7; DEV = C.DEVICE; SEED = 999
MAX_TOK = 1100
P1, P2 = 2, 8   # mismo número que el run original (matching aproximado)
CKPT_ORIG = os.path.join(ALT, "vista_e_task23_longformer_best.pt")
CKPT_FULL = os.path.join(ALT, "vista_e_task23_longformer_full_best.pt")

# Reutiliza el pipeline del max512 (load, DS, loss, etc.)
from task23_max512 import load_t23, DS23, cat_posw, loss_fn, gold_hard, pred_hard, pyevall_hard, pyevall_soft


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
                    eeg=eeg, eeg_mask=mask, soft=torch.stack([x["soft"] for x in b]), sex=torch.stack([x["sex"] for x in b]))
    return f


class VistaELong23(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(LONG_MODEL, torch_dtype=C.AMP_DTYPE)
        try: self.text_model.gradient_checkpointing_enable()
        except Exception: pass
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS + 6
        self.trunk = nn.Sequential(nn.Linear(fd, 512), nn.GELU(), nn.Dropout(C.DROPOUT),
                                   nn.Linear(512, 256), nn.GELU(), nn.Dropout(C.DROPOUT))
        self.head_sex = nn.Linear(256, 1)
        self.head_cat = nn.Linear(256, 5)
    def forward(self, b):
        out = self.text_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                              global_attention_mask=b.get("global_attention_mask"))
        last = out.last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(last.dtype)
        t = ((last * m).sum(1) / m.sum(1).clamp(min=1e-9)).float()
        e, _ = self.eeg_pool(b["eeg"].float(), b["eeg_mask"])
        x = torch.cat([t, e, b["emotions"].float(), b["feat"].float()], dim=1)
        h = self.trunk(x)
        return self.head_sex(h).squeeze(-1), self.head_cat(h)


@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, PS, PC, T, SX = [], [], [], [], []
    for b in dl:
        bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            ls, lc = model(bb)
        ids += b["id"]; PS.append(torch.sigmoid(ls).float().cpu().numpy())
        PC.append(torch.sigmoid(lc).float().cpu().numpy())
        T.append(b["soft"].numpy()); SX.append(b["sex"].numpy())
    return ids, np.concatenate(PS), np.concatenate(PC), np.concatenate(T), np.concatenate(SX)


def search_thresholds(model, dl_va):
    """Busca (tsex, tcat) en val maximizando 0.5*F1 + 0.5*ICMNorm."""
    ids, ps, pc, T, SX = infer(model, dl_va)
    gh = gold_hard(T, SX)
    best_sc, best = -1, (0.5, np.full(5, 0.5))
    for tsex in np.arange(0.30, 0.66, 0.04):
        for tc in np.arange(0.10, 0.55, 0.05):
            pr = pred_hard(ps, pc, tsex, np.full(5, tc))
            icm, icmn, fm = pyevall_hard(ids, gh, pr)
            sc = 0.5 * fm + 0.5 * icmn
            if sc > best_sc: best_sc, best = sc, (float(tsex), np.full(5, float(tc)))
    tsex, tcat = best
    icm, icmn, fm = pyevall_hard(ids, gh, pred_hard(ps, pc, tsex, tcat))
    s0 = pyevall_soft(ids, T, SX, ps, pc)
    print(f"  [val] tsex={tsex:.2f} tcat={tcat[0]:.2f}  ICM={icm:+.4f} F1={fm:.4f}  ICMSoft={s0[0]:+.4f}", flush=True)
    return tsex, tcat


def main():
    C.set_seed(SEED); C.configure_gpu()
    print(f"[Task 2.3 Longformer FULL] {LONG_MODEL} MAX_TOK={MAX_TOK}", flush=True)
    splits = load_t23()
    tok = AutoTokenizer.from_pretrained(LONG_MODEL); cl = collate(tok)

    # ====== PASO 1: cargar modelo original y buscar thresholds en val ======
    print("\n[PASO 1] Cargando checkpoint Longformer original y buscando thresholds en val...", flush=True)
    dl_va_only = DataLoader(DS23(splits["val"]), batch_size=4, shuffle=False, collate_fn=cl, num_workers=4)
    model_orig = VistaELong23().to(DEV)
    if os.path.exists(CKPT_ORIG):
        sd = torch.load(CKPT_ORIG, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        model_orig.load_state_dict(sd)
        print(f"  modelo original cargado de {CKPT_ORIG}", flush=True)
        tsex_orig, tcat_orig = search_thresholds(model_orig, dl_va_only)
        # Guardamos los thresholds del modelo original
        json.dump({"tsex": float(tsex_orig), "tcat": tcat_orig.tolist()},
                  open(os.path.join(ALT, "task23_longformer_thresholds_val.json"), "w"))
    else:
        print(f"  ⚠️ no existe {CKPT_ORIG} — usaré thresholds por defecto", flush=True)
        tsex_orig, tcat_orig = 0.50, np.full(5, 0.30)

    del model_orig; torch.cuda.empty_cache()

    # ====== PASO 2: reentrenar con train+val combinados ======
    print(f"\n[PASO 2] Reentrenando con train+val combinados (n={len(splits['train']) + len(splits['val'])})...", flush=True)
    full_train = splits["train"] + splits["val"]
    dl_tr = DataLoader(DS23(full_train), batch_size=4, shuffle=True, collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS23(splits["val"]), batch_size=8, shuffle=False, collate_fn=cl, num_workers=4)  # solo para monitorizar
    posw = cat_posw({"train": full_train})
    model = VistaELong23().to(DEV)

    def run_epoch(opt, sch):
        model.train()
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                ls, lc = model(bb); loss = loss_fn(ls, lc, bb["soft"], bb["sex"], posw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

    def quick_log(tag):
        ids, ps, pc, T, SX = infer(model, dl_va)
        yb = (T > 1/6 + 1e-9).astype(int)
        pb = (pc >= 0.5).astype(int) * (SX[:, None] >= 0.5)
        f1 = f1_score(yb.ravel(), pb.ravel())
        print(f"  [{tag}] F1(cat micro, val-in-train)={f1:.4f}", flush=True)

    # Fase 1: backbone congelado
    for p in model.text_model.parameters(): p.requires_grad = False
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"  FASE 1 ({P1} ép, {sum(p.numel() for p in tr):,} params)", flush=True)
    opt = torch.optim.AdamW(tr, lr=5e-5, weight_decay=0.01)
    st = max(1, len(dl_tr) * P1); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    for e in range(1, P1 + 1): run_epoch(opt, sch); quick_log(f"F1 {e}/{P1}")
    del opt, sch; torch.cuda.empty_cache()

    # Fase 2: full fine-tune, sin early stopping, N épocas fijas
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

    # Guardar checkpoint final
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(model_state_dict=sd), CKPT_FULL)
    print(f"  checkpoint final -> {CKPT_FULL}", flush=True)

    # ====== PASO 3: inferencia sobre test y submission ======
    print(f"\n[PASO 3] Inferencia sobre test con modelo retrained, aplicando thresholds del PASO 1...", flush=True)
    dl_te = DataLoader(DS23(splits["test"]), batch_size=4, shuffle=False, collate_fn=cl, num_workers=4)
    tids, tps, tpc, _, _ = infer(model, dl_te)
    print(f"  test n={len(tids)}, tsex={tsex_orig:.3f}, tcat={tcat_orig[0]:.3f}", flush=True)

    # ====== PASO 4: escribir submissions task2_3 ======
    OUT = C.OUT_DIR
    hard_preds = pred_hard(tps, tpc, tsex_orig, tcat_orig)
    # Soft: combinar prob(sex) y prob(cat)
    soft_combined = tps[:, None] * tpc        # (n, 5)
    no_prob = 1.0 - tps                        # (n,)
    def sval(p_no, c5):
        return {"NO": float(max(0.0, p_no)),
                **{CATS[c]: float(c5[c]) for c in range(5)}}

    # Para Task 2.3 mantenemos 3 submissions con variaciones de threshold:
    #   1) thresholds originales (champion)
    #   2) mismos thresholds (variante)
    #   3) thresholds +0.05 (más conservadora)
    runs = {
        "1": (tsex_orig, tcat_orig),
        "2": (tsex_orig, tcat_orig),
        "3": (min(tsex_orig + 0.05, 0.9), np.clip(tcat_orig + 0.05, 0.05, 0.9)),
    }
    for n, (tsex, tcat) in runs.items():
        hard = pred_hard(tps, tpc, tsex, tcat)
        soft_out = []
        hard_out = []
        for i, mid in enumerate(tids):
            hard_out.append({"test_case": TC, "id": str(mid), "value": hard[i]})
            soft_out.append({"test_case": TC, "id": str(mid),
                             "value": sval(no_prob[i], soft_combined[i])})
        json.dump(hard_out, open(os.path.join(OUT, f"task2_3_hard_{C.TEAM_NAME}_{n}"), "w"), indent=2)
        json.dump(soft_out, open(os.path.join(OUT, f"task2_3_soft_{C.TEAM_NAME}_{n}"), "w"), indent=2)
        print(f"  escrito task2_3_{{hard,soft}}_{C.TEAM_NAME}_{n}  (tsex={tsex:.3f} tcat={tcat[0]:.3f})", flush=True)

    # ====== PASO 5: rezip ======
    print(f"\n[PASO 5] Regenerando zip final...", flush=True)
    import zipfile
    fns = sorted(f for f in os.listdir(OUT) if f.startswith("task2_"))
    zp = os.path.join(C.ROOT, f"exist2026_{C.TEAM_NAME}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in fns:
            zf.write(os.path.join(OUT, fn), arcname=os.path.join(f"exist2026_{C.TEAM_NAME}", fn))
    print(f"  ZIP: {zp}  ({os.path.getsize(zp)/1024:.1f} KB, {len(fns)} ficheros)", flush=True)
    print("\n=== TERMINADO Task 2.3 Longformer FULL ===", flush=True)


if __name__ == "__main__":
    main()
