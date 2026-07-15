"""EXPERIMENTO B — Task 2.3 con XLM-R-Longformer-base-4096. Multi-label 5 cats.
Sin warm-start (arquitectura distinta). max_length=1100."""
import json, os, tempfile
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import config as C
import data as D
from models import SetAttentionPool
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
LONG_MODEL = "markussagen/xlm-roberta-longformer-base-4096"
CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]
ALL = ["NO"] + CATS; HIER = {"YES": CATS, "NO": []}
TC = "EXIST2025"; EPS = 1e-7; DEV = C.DEVICE; SEED = 999
MAX_TOK = 1100
P1, P2, PAT = 2, 8, 3
CKPT = os.path.join(ALT, "vista_e_task23_longformer_best.pt")

# reutilizar load_t23, DS23, gold_hard, pred_hard, pyevall_* del max512
from task23_max512 import load_t23, DS23, cat_posw, loss_fn, gold_hard, pred_hard, pyevall_hard, pyevall_soft


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j, :n] = x["eeg"]; mask[j, :n] = True
        gam = torch.zeros_like(enc["input_ids"]); gam[:, 0] = 1
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], global_attention_mask=gam,
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
        self.trunk = nn.Sequential(nn.Linear(fd, 512), nn.GELU(), nn.Dropout(C.DROPOUT), nn.Linear(512, 256), nn.GELU(), nn.Dropout(C.DROPOUT))
        self.head_sex = nn.Linear(256, 1)
        self.head_cat = nn.Linear(256, 5)

    def forward(self, b):
        out = self.text_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], global_attention_mask=b.get("global_attention_mask"))
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
        ids += b["id"]; PS.append(torch.sigmoid(ls).float().cpu().numpy()); PC.append(torch.sigmoid(lc).float().cpu().numpy())
        T.append(b["soft"].numpy()); SX.append(b["sex"].numpy())
    return ids, np.concatenate(PS), np.concatenate(PC), np.concatenate(T), np.concatenate(SX)


def main():
    C.set_seed(SEED); C.configure_gpu()
    print(f"[EXP-B 2.3 Longformer] {LONG_MODEL} MAX_TOK={MAX_TOK}", flush=True)
    splits = load_t23()
    tok = AutoTokenizer.from_pretrained(LONG_MODEL); cl = collate(tok)
    dl_tr = DataLoader(DS23(splits["train"]), batch_size=4, shuffle=True, collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS23(splits["val"]), batch_size=8, shuffle=False, collate_fn=cl, num_workers=4)
    posw = cat_posw(splits)
    model = VistaELong23().to(DEV)

    def run_epoch(opt, sch):
        model.train()
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                ls, lc = model(bb); loss = loss_fn(ls, lc, bb["soft"], bb["sex"], posw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

    def ev(tag):
        ids, ps, pc, T, SX = infer(model, dl_va)
        yb = (T > 1/6 + 1e-9).astype(int); pb = (pc >= 0.5).astype(int) * (SX[:, None] >= 0.5)
        f1 = f1_score(yb.ravel(), pb.ravel())
        print(f"  [{tag}] val F1(cat micro)={f1:.4f}", flush=True); return f1

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
        run_epoch(opt, sch); f1 = ev(f"F2 {e}/{P2}")
        if f1 > best: best, bsd, pat = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0; print(f"     mejor -> ckpt ({f1:.4f})", flush=True)
        else:
            pat += 1
            if pat >= PAT: print("     early stopping", flush=True); break
    if bsd: model.load_state_dict(bsd)
    torch.save(dict(model_state_dict=bsd, val_f1=best), CKPT)

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
    print(f"\n=== RESULTADOS Task 2.3 EXP-B (Longformer max_length={MAX_TOK}) ===", flush=True)
    print(f"  thr_sex={tsex:.2f}  thr_cat={tcat[0]:.2f}", flush=True)
    print(f"  HARD: ICM={icm:+.4f} ICMNorm={icmn:.3f} F1macro={fm:.4f}", flush=True)
    print(f"  SOFT: ICMSoft={s0[0]:+.4f} ICMSoftNorm={s0[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
