"""EXPERIMENTO B — Task 2.1 con XLM-R-Longformer-base-4096 (multilingüe, 4096 tokens).
Texto enriquecido + EEG + Ekman. Binary classification. NO warm-start (arquitectura distinta)."""
import json, os, tempfile
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
import config as C
import data as D
from models import SetAttentionPool
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
LONG_MODEL = "markussagen/xlm-roberta-longformer-base-4096"
MAX_TOK = 1100; TC = "EXIST2025"; EPS = 1e-7; DEV = C.DEVICE; SEED = 999
P1, P2, PAT = 2, 8, 3
CKPT = os.path.join(ALT, "vista_e_task21_longformer_best.pt")


def load_caps():
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    caps = {}
    for mid, v in g.items():
        if isinstance(v, dict):
            d = (v.get("description") or "").strip(); a = (v.get("sexism_analysis") or "").strip()
            parts = []
            if d: parts.append(f"Description: {d}")
            if a: parts.append(f"Sexism Analysis: {a}")
            if parts: caps[str(mid)] = " ".join(parts)
    return caps


class DS21(Dataset):
    def __init__(self, ex, caps): self.ex, self.caps = ex, caps
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]
        text = (self.caps.get(e["id"], "") + " " + e["text"]).strip()
        return dict(id=e["id"], text=text,
                    emotions=torch.from_numpy(np.asarray(e.get("emotions", np.zeros(C.N_EMOTIONS)), np.float32)),
                    eeg=torch.from_numpy(e["sensors_z"]["EEG"]),
                    soft=torch.tensor(float(e["soft"]) if e["soft"] is not None else -1.0, dtype=torch.float32))


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j, :n] = x["eeg"]; mask[j, :n] = True
        gam = torch.zeros_like(enc["input_ids"]); gam[:, 0] = 1
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], global_attention_mask=gam,
                    emotions=torch.stack([x["emotions"] for x in b]), eeg=eeg, eeg_mask=mask,
                    soft=torch.stack([x["soft"] for x in b]))
    return f


class VistaELong21(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(LONG_MODEL, torch_dtype=C.AMP_DTYPE)
        try: self.text_model.gradient_checkpointing_enable()
        except Exception: pass
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS
        self.head = nn.Sequential(nn.Dropout(C.DROPOUT), nn.Linear(fd, 512), nn.GELU(), nn.Dropout(C.DROPOUT), nn.Linear(512, 1))

    def forward(self, b):
        out = self.text_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], global_attention_mask=b.get("global_attention_mask"))
        last = out.last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(last.dtype)
        t = ((last * m).sum(1) / m.sum(1).clamp(min=1e-9)).float()
        e, _ = self.eeg_pool(b["eeg"].float(), b["eeg_mask"])
        x = torch.cat([t, e, b["emotions"].float()], dim=1)
        return self.head(x).squeeze(-1)


@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, lo, tg = [], [], []
    for b in dl:
        bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(bb)
        ids += b["id"]; lo.append(logit.float().cpu().numpy()); tg.append(b["soft"].numpy())
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
    print(f"[EXP-B 2.1 Longformer] {LONG_MODEL} MAX_TOK={MAX_TOK}", flush=True)
    splits = D.load_split(); caps = load_caps()
    tok = AutoTokenizer.from_pretrained(LONG_MODEL); cl = collate(tok)
    dl_tr = DataLoader(DS21(splits["train"], caps), batch_size=4, shuffle=True, drop_last=False, collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS21(splits["val"], caps), batch_size=8, shuffle=False, collate_fn=cl, num_workers=4)
    model = VistaELong21().to(DEV)

    def run_epoch(opt, sch):
        model.train()
        pos_w = torch.tensor([C.POS_WEIGHT], device=DEV)
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                logit = model(bb); loss = F.binary_cross_entropy_with_logits(logit, bb["soft"], pos_weight=pos_w)
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

    ids, lo, tg = infer(model, dl_va)
    probs = 1 / (1 + np.exp(-lo))
    y = (tg >= 0.5).astype(int)
    best_thr, best_icm = 0.5, -1e9
    for t in [round(0.30 + 0.01 * i, 2) for i in range(41)]:
        ic = icm_hard(ids, y, (probs >= t).astype(int))
        if ic > best_icm: best_icm, best_thr = ic, t
    auc = roc_auc_score(y, probs); f1 = f1_score(y, probs >= best_thr)
    icm_s = icmsoft(ids, tg, probs)
    print(f"\n=== RESULTADOS Task 2.1 EXP-B (Longformer max_length={MAX_TOK}) ===", flush=True)
    print(f"  thr óptimo={best_thr}", flush=True)
    print(f"  HARD: ICM={best_icm:+.4f}  F1+={f1:.4f}  AUC={auc:.4f}", flush=True)
    print(f"  SOFT: ICMSoft={icm_s:+.4f}", flush=True)
    print("  Modelo actual XLM-R-base (max=256): ICM +0.386 · F1+ 0.861 · AUC 0.884 · ICMSoft +0.480", flush=True)


if __name__ == "__main__":
    main()
