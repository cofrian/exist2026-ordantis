"""EXPERIMENTO A — Task 2.1 con XLM-R-base a max_length=512 (antes 256).
Vista E-2.1: texto enriquecido (OCR + descripción Gemini + análisis) + EEG + Ekman.
Warm-start del checkpoint actual M3_vista_E_best.pt. Guarda en _alt/, no toca el zip."""
import json, os, tempfile
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import config as C
import data as D
from models import SetAttentionPool, encode_text_mean_pool
from dataset import MemeDataset, make_collate, to_device
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
TC = "EXIST2025"; EPS = 1e-7; DEV = C.DEVICE; SEED = 999
MAX_TOK = 512   # ← clave
P1, P2, PAT = 5, 15, 4
CKPT = os.path.join(ALT, "vista_e_task21_max512_R_best.pt")


def load_caps():
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    caps = {}
    for mid, v in g.items():
        if isinstance(v, dict):
            d = (v.get("description") or "").strip(); a = (v.get("sexism_analysis") or "").strip()
            parts = []
            r = (v.get("reasoning") or "").strip()
            if d: parts.append(f"Description: {d}")
            if a: parts.append(f"Sexism Analysis: {a}")
            if r: parts.append(f"Reasoning: {r}")
            if parts: caps[str(mid)] = " ".join(parts)
    return caps


class VistaE21(nn.Module):
    """Vista E-2.1: text (mean-pool) + EEG (set-pool) + Ekman → binary."""
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(C.TEXT_MODEL, torch_dtype=C.AMP_DTYPE, attn_implementation=C.best_attn_impl())
        self.text_model.gradient_checkpointing_enable()
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS
        self.head = nn.Sequential(nn.Dropout(C.DROPOUT), nn.Linear(fd, 512), nn.GELU(),
                                  nn.Dropout(C.DROPOUT), nn.Linear(512, 1))

    def forward(self, batch):
        t = encode_text_mean_pool(self.text_model, batch["input_ids"], batch["attention_mask"]).float()
        e, _ = self.eeg_pool(batch["sens_EEG"].float(), batch["mask_EEG"])
        x = torch.cat([t, e, batch["emotions"].float()], dim=1)
        return self.head(x).squeeze(-1)


def warm_start(model):
    p = os.path.join(C.CKPT_DIR, "M3_vista_E_best.pt")
    if not os.path.exists(p): return
    sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    tm = {k[len("text_model."):]: v for k, v in sd.items() if k.startswith("text_model.")}
    model.text_model.load_state_dict(tm, strict=False)
    ep = {k[len("pools.EEG."):]: v for k, v in sd.items() if k.startswith("pools.EEG.")}
    if ep: model.eeg_pool.load_state_dict(ep, strict=False)
    # head: la del checkpoint es Sequential idéntica
    hd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    try: model.head.load_state_dict(hd, strict=False)
    except Exception: pass
    print(f"  [warm-start] XLM-R {len(tm)} + EEG {len(ep)} + head {len(hd)}", flush=True)


@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, lo, tg = [], [], []
    for b in dl:
        b = to_device(b, DEV)
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(b)
        ids += b["id"]; lo.append(logit.float().cpu().numpy()); tg.append(b["soft"].cpu().numpy())
    return ids, np.concatenate(lo), np.concatenate(tg)


def icm_hard(ids, y, pred):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td, "p"), os.path.join(td, "g")
        json.dump([{"test_case": TC, "id": str(i), "value": ("YES" if pred[k] else "NO")} for k, i in enumerate(ids)], open(pf, "w"))
        json.dump([{"test_case": TC, "id": str(i), "value": ("YES" if y[k] else "NO")} for k, i in enumerate(ids)], open(gf, "w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICM"], **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED, PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        return rep.report["metrics"]["ICM"]["results"]["average_per_test_case"]


def icmsoft(ids, soft_gold, probs):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td, "p"), os.path.join(td, "g")
        json.dump([{"test_case": TC, "id": str(i), "value": {"YES": float(probs[k]), "NO": float(1 - probs[k])}} for k, i in enumerate(ids)], open(pf, "w"))
        json.dump([{"test_case": TC, "id": str(i), "value": {"YES": float(soft_gold[k]), "NO": float(1 - soft_gold[k])}} for k, i in enumerate(ids)], open(gf, "w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICMSoft"], **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED, PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        return rep.report["metrics"]["ICMSoft"]["results"]["average_per_test_case"]


def main():
    C.set_seed(SEED); C.configure_gpu()
    print(f"[EXP-A 2.1 max512+REASONING] MAX_TOK={MAX_TOK}", flush=True)
    splits = D.load_split()
    caps = load_caps()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    # MemeDataset usa C.MAX_TOKENS, no MAX_TOK; lo subimos:
    C.MAX_TOKENS = MAX_TOK
    coll = make_collate(tok)
    vit_emb = {e["id"]: np.zeros(768, dtype=np.float32) for e in splits["train"] + splits["val"] + splits["test"]}
    ds_tr = MemeDataset(splits["train"], tok, vit_emb=vit_emb, captions=caps, use_caption=True)
    ds_va = MemeDataset(splits["val"], tok, vit_emb=vit_emb, captions=caps, use_caption=True)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, drop_last=False, collate_fn=coll, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=32, shuffle=False, collate_fn=coll, num_workers=4)
    model = VistaE21().to(DEV); warm_start(model)

    def run_epoch(opt, sch):
        model.train()
        pos_w = torch.tensor([C.POS_WEIGHT], device=DEV)
        for b in dl_tr:
            b = to_device(b, DEV)
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                logit = model(b); loss = F.binary_cross_entropy_with_logits(logit, b["soft"], pos_weight=pos_w)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

    def ev(tag):
        ids, lo, tg = infer(model, dl_va)
        probs = 1 / (1 + np.exp(-lo))
        y = (tg >= 0.5).astype(int); pred = (probs >= 0.5).astype(int)
        f1 = f1_score(y, pred); auc = roc_auc_score(y, probs)
        print(f"  [{tag}] val F1+={f1:.4f}  AUC={auc:.4f}", flush=True)
        return f1, auc

    for p in model.text_model.parameters(): p.requires_grad = False
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"FASE 1 ({P1} ép, {sum(p.numel() for p in tr):,} params)", flush=True)
    opt = torch.optim.AdamW(tr, lr=5e-5, weight_decay=0.01)
    st = max(1, len(dl_tr) * P1); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    for e in range(1, P1 + 1): run_epoch(opt, sch); ev(f"F1 {e}/{P1}")
    del opt, sch; torch.cuda.empty_cache()

    for p in model.text_model.parameters(): p.requires_grad = True
    low, high = [], []
    for n, p in model.text_model.named_parameters():
        ln = int(n.split("encoder.layer.")[1].split(".")[0]) if "encoder.layer." in n else None
        (low if (("embeddings" in n) or (ln is not None and ln <= 6)) else high).append(p)
    headp = [p for n, p in model.named_parameters() if not n.startswith("text_model.")]
    opt = torch.optim.AdamW([{"params": low, "lr": 1e-5}, {"params": high, "lr": 3e-5}, {"params": headp, "lr": 1e-4}], weight_decay=0.01)
    st = max(1, len(dl_tr) * P2); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    print(f"FASE 2 ({P2} ép)", flush=True)
    best, bsd, pat = -1, None, 0
    for e in range(1, P2 + 1):
        run_epoch(opt, sch); f1, _ = ev(f"F2 {e}/{P2}")
        if f1 > best: best, bsd, pat = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0; print(f"     mejor -> ckpt ({f1:.4f})", flush=True)
        else:
            pat += 1
            if pat >= PAT: print("     early stopping", flush=True); break
    if bsd: model.load_state_dict(bsd)
    torch.save(dict(model_state_dict=bsd, val_f1=best), CKPT)

    # eval final
    ids, lo, tg = infer(model, dl_va)
    probs = 1 / (1 + np.exp(-lo))
    y = (tg >= 0.5).astype(int)
    # threshold óptimo
    best_thr, best_icm = 0.5, -1e9
    for t in [round(0.30 + 0.01 * i, 2) for i in range(41)]:
        ic = icm_hard(ids, y, (probs >= t).astype(int))
        if ic > best_icm: best_icm, best_thr = ic, t
    auc = roc_auc_score(y, probs); f1 = f1_score(y, probs >= best_thr)
    icm_s = icmsoft(ids, tg, probs)
    print(f"\n=== RESULTADOS Task 2.1 EXP-A (max_length={MAX_TOK}) ===", flush=True)
    print(f"  thr óptimo={best_thr}", flush=True)
    print(f"  HARD: ICM={best_icm:+.4f}  F1+={f1:.4f}  AUC={auc:.4f}", flush=True)
    print(f"  SOFT: ICMSoft={icm_s:+.4f}", flush=True)
    print("  Modelo actual (max_length=256): ICM +0.386 · F1+ 0.861 · AUC 0.884 · ICMSoft +0.480", flush=True)


if __name__ == "__main__":
    main()
