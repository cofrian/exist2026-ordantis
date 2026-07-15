"""EXPERIMENTO A — Task 2.3 con XLM-R-base a max_length=512. Multi-label 5 categorías.
Warm-start desde Vista E-2.2 (compatible). Guarda en _alt/."""
import json, os, tempfile
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import config as C
import data as D
from models import SetAttentionPool, encode_text_mean_pool
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]
ALL = ["NO"] + CATS; HIER = {"YES": CATS, "NO": []}
TC = "EXIST2025"; EPS = 1e-7; DEV = C.DEVICE; SEED = 999
MAX_TOK = 512   # ← clave
P1, P2, PAT = 3, 12, 4
CKPT = os.path.join(ALT, "vista_e_task23_max512_R_best.pt")


def load_t23():
    splits = D.load_split()
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    raw = json.load(open(C.TRAIN_JSON, encoding="utf-8"))
    soft, sexp = {}, {}
    for m in raw.values():
        ann = [a for a in m["labels_task2_3"] if a != "UNKNOWN"]
        n = len(ann) or 1
        nsex = sum(1 for a in ann if a != "-")
        cnt = {c: sum(1 for a in ann if isinstance(a, list) and c in a) for c in CATS}
        soft[str(m["id_EXIST"])] = [cnt[c] / n for c in CATS]
        sexp[str(m["id_EXIST"])] = nsex / n
    def gd(mid):
        v = g.get(str(mid)); return v if isinstance(v, dict) else None
    def enrich(e):
        d = gd(e["id"]); ocr = e["text"]
        if not d: return ocr, np.zeros(6, np.float32)
        desc = (d.get("description") or "").strip()
        t23 = d.get("task2_3", {}) or {}
        cr = (t23.get("category_reasoning") or "").strip()
        sa = (d.get("sexism_analysis") or "").strip()
        present = ", ".join(c for c in (t23.get("categories_present") or []) if c in CATS) or "none"
        rsn = (d.get("reasoning") or "").strip()
        txt = f"{ocr} </s> {desc} </s> CATEGORIES: {present} </s> {cr} </s> {sa} </s> {rsn}"
        cp = t23.get("category_probabilities", {}) or {}
        sp = float(d.get("task2_1", {}).get("sexist_probability", 0.0) or 0.0)
        feat = np.array([sp] + [float(cp.get(c, 0.0) or 0.0) for c in CATS], np.float32)
        return txt, feat
    for part in ("train", "val", "test"):
        for e in splits[part]:
            e["t23_text"], e["t23_feat"] = enrich(e)
            e["t23_soft"] = soft.get(e["id"]); e["t23_sex"] = sexp.get(e["id"])
    return splits


class DS23(Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]
        return dict(id=e["id"], text=e["t23_text"], feat=torch.from_numpy(e["t23_feat"]),
                    emotions=torch.from_numpy(np.asarray(e.get("emotions", np.zeros(C.N_EMOTIONS)), np.float32)),
                    eeg=torch.from_numpy(e["sensors_z"]["EEG"]),
                    soft=torch.tensor(e["t23_soft"] if e["t23_soft"] is not None else [-1]*5, dtype=torch.float32),
                    sex=torch.tensor(e["t23_sex"] if e["t23_sex"] is not None else -1.0, dtype=torch.float32))


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j, :n] = x["eeg"]; mask[j, :n] = True
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    feat=torch.stack([x["feat"] for x in b]), emotions=torch.stack([x["emotions"] for x in b]),
                    eeg=eeg, eeg_mask=mask, soft=torch.stack([x["soft"] for x in b]), sex=torch.stack([x["sex"] for x in b]))
    return f


class VistaE23(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(C.TEXT_MODEL, torch_dtype=C.AMP_DTYPE, attn_implementation=C.best_attn_impl())
        self.text_model.gradient_checkpointing_enable()
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS + 6
        self.trunk = nn.Sequential(nn.Linear(fd, 512), nn.GELU(), nn.Dropout(C.DROPOUT), nn.Linear(512, 256), nn.GELU(), nn.Dropout(C.DROPOUT))
        self.head_sex = nn.Linear(256, 1)
        self.head_cat = nn.Linear(256, 5)

    def forward(self, b):
        t = encode_text_mean_pool(self.text_model, b["input_ids"], b["attention_mask"]).float()
        e, _ = self.eeg_pool(b["eeg"].float(), b["eeg_mask"])
        x = torch.cat([t, e, b["emotions"].float(), b["feat"].float()], dim=1)
        h = self.trunk(x)
        return self.head_sex(h).squeeze(-1), self.head_cat(h)


def warm_start(model):
    p = os.path.join(C.CKPT_DIR, "vista_e_task22_best.pt")
    if not os.path.exists(p): p = os.path.join(C.CKPT_DIR, "M3_vista_E_best.pt")
    if not os.path.exists(p): return
    sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    tm = {k[len("text_model."):]: v for k, v in sd.items() if k.startswith("text_model.")}
    model.text_model.load_state_dict(tm, strict=False)
    ep = {k[len("pools.EEG."):]: v for k, v in sd.items() if k.startswith("pools.EEG.")}
    if not ep:
        ep = {k[len("eeg_pool."):]: v for k, v in sd.items() if k.startswith("eeg_pool.")}
    if ep: model.eeg_pool.load_state_dict(ep, strict=False)
    print(f"  [warm-start] desde {os.path.basename(p)}: XLM-R {len(tm)} + EEG {len(ep)}", flush=True)


def cat_posw(splits):
    M = np.array([e["t23_soft"] for e in splits["train"] if e["t23_soft"] is not None])
    pos = M.mean(0).clip(0.02, 0.98)
    return torch.tensor(((1 - pos) / pos), dtype=torch.float32).clamp(0.5, 8.0)



def sampler_for(ex):
    # Sobre-muestrea memes con categorías minoritarias (SEX-VIOL / MISO-NSV)
    weights = []
    for e in ex:
        s = e.get("t23_soft") or [0]*5
        w = 1.0 + 2.0 * float(s[3] >= 0.34) + 2.0 * float(s[4] >= 0.34)
        weights.append(w)
    return WeightedRandomSampler(torch.tensor(weights), len(weights), replacement=True)

def loss_fn(ls, lc, soft, sex, posw, gamma=2.0):
    tgt_sex = sex.clamp(0, 1)
    Ls = F.binary_cross_entropy_with_logits(ls, tgt_sex)
    den = tgt_sex.clamp(min=1e-6).unsqueeze(1)
    tgt_cat = (soft / den).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(lc, tgt_cat, pos_weight=posw.to(lc.device), reduction="none")
    p = torch.sigmoid(lc); pt = p * tgt_cat + (1 - p) * (1 - tgt_cat)
    fl = ((1 - pt).pow(gamma) * bce).mean(1)
    m = (tgt_sex > 0.5).float()
    Lc = (fl * m).sum() / m.sum().clamp(min=1)
    return Ls + Lc


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


def gold_hard(soft5, sex):
    y = []
    for i in range(len(sex)):
        if sex[i] < 0.5: y.append(["NO"]); continue
        cats = [CATS[c] for c in range(5) if soft5[i, c] > (1.0 / 6 + 1e-9)]
        y.append(cats if cats else [CATS[int(np.argmax(soft5[i]))]])
    return y


def pred_hard(ps, pc, thr_sex, thr_cat):
    out = []
    for i in range(len(ps)):
        if ps[i] < thr_sex: out.append(["NO"]); continue
        cats = [CATS[c] for c in range(5) if pc[i, c] >= thr_cat[c]]
        out.append(cats if cats else [CATS[int(np.argmax(pc[i]))]])
    return out


def pyevall_hard(ids, gold, pred):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td, "p"), os.path.join(td, "g")
        json.dump([{"test_case": TC, "id": str(i), "value": pred[k]} for k, i in enumerate(ids)], open(pf, "w"))
        json.dump([{"test_case": TC, "id": str(i), "value": gold[k]} for k, i in enumerate(ids)], open(gf, "w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICM", "ICMNorm", "FMeasure"], **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED, PyEvALLUtils.PARAM_HIERARCHY: HIER, PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        m = rep.report["metrics"]
        return (m["ICM"]["results"]["average_per_test_case"], m["ICMNorm"]["results"]["average_per_test_case"],
                m["FMeasure"]["results"]["test_cases"][0]["average"])


def pyevall_soft(ids, soft5, sex, ps, pc):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td, "p"), os.path.join(td, "g")
        def sval(p_no, c5): return {"NO": float(max(0.0, p_no)), **{CATS[c]: float(c5[c]) for c in range(5)}}
        json.dump([{"test_case": TC, "id": str(i), "value": sval(1 - ps[k], ps[k] * pc[k])} for k, i in enumerate(ids)], open(pf, "w"))
        json.dump([{"test_case": TC, "id": str(i), "value": sval(1 - sex[k], soft5[k])} for k, i in enumerate(ids)], open(gf, "w"))
        try:
            rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICMSoft", "ICMSoftNorm"], **{PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED, PyEvALLUtils.PARAM_HIERARCHY: HIER, PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
            m = rep.report["metrics"]
            return m["ICMSoft"]["results"]["average_per_test_case"], m["ICMSoftNorm"]["results"]["average_per_test_case"]
        except Exception as ex:
            print(f"  [aviso] ICMSoft falló ({ex}); devuelvo NaN", flush=True); return float("nan"), float("nan")


def main():
    C.set_seed(SEED); C.configure_gpu()
    print(f"[EXP-A 2.3 max512+SAMPLER+REASONING] MAX_TOK={MAX_TOK}", flush=True)
    splits = load_t23()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL); cl = collate(tok)
    dl_tr = DataLoader(DS23(splits["train"]), batch_size=8, sampler=sampler_for(splits["train"]), collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS23(splits["val"]), batch_size=32, shuffle=False, collate_fn=cl, num_workers=4)
    posw = cat_posw(splits); print("  pos_weight:", [round(float(x), 2) for x in posw], flush=True)
    model = VistaE23().to(DEV); warm_start(model)

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
    print(f"\n=== RESULTADOS Task 2.3 EXP-A (max_length={MAX_TOK}) ===", flush=True)
    print(f"  thr_sex={tsex:.2f}  thr_cat={tcat[0]:.2f}", flush=True)
    print(f"  HARD: ICM={icm:+.4f} ICMNorm={icmn:.3f} F1macro={fm:.4f}", flush=True)
    print(f"  SOFT: ICMSoft={s0[0]:+.4f} ICMSoftNorm={s0[1]:.3f}", flush=True)
    print("  Modelo actual (max_length=320): F1macro 0.581 · ICMSoft-proxy -0.383", flush=True)


if __name__ == "__main__":
    main()
