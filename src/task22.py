"""Task 2.2 (intención del autor: NO / DIRECT / JUDGEMENTAL) — Vista E adaptada.

Mejoras: head jerárquico (binary + type), soft labels (distribución de los 6 anotadores),
Weighted Random Sampler (sobre-muestrea JUDGEMENTAL), Focal Loss para el tipo,
warm-start del XLM-R desde Vista E de Task 2.1, texto enriquecido con campos de Gemini,
threshold 2D sobre validación. Genera 3 runs hard + 3 soft y re-empaqueta el zip.

NO re-promptea Gemini: usa los campos ya guardados en gemini_predictions.json.
"""
import json, os, math, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

import config as C
import data as D
from models import SetAttentionPool, encode_text_mean_pool

INT = ["NO", "DIRECT", "JUDGEMENTAL"]
DEV = C.DEVICE
SEED = 999
P1_EPOCHS, P2_EPOCHS, PATIENCE = 3, 10, 4
MAX_TOK = 320


# ---------------------------------------------------------------- datos
def load_task22():
    splits = D.load_split()
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    train_raw = json.load(open(C.TRAIN_JSON, encoding="utf-8"))
    # distribución soft y mayoritaria por meme (sobre los 6 anotadores)
    soft22, maj22 = {}, {}
    for k, m in train_raw.items():
        votes = [("NO" if v == "-" else v) for v in m["labels_task2_2"] if v != "UNKNOWN"]
        c = {x: votes.count(x) for x in INT}
        n = sum(c.values())
        soft22[str(m["id_EXIST"])] = ([c[x] / n for x in INT] if n else [1.0, 0.0, 0.0])
        maj22[str(m["id_EXIST"])] = (max(c, key=c.get) if n else "NO")

    def gd(mid):
        v = g.get(str(mid))
        return v if isinstance(v, dict) else None

    def enrich(e):
        d = gd(e["id"])
        ocr = e["text"]
        if not d:
            return ocr, np.zeros(7, np.float32)
        desc = (d.get("description") or "").strip()
        sa = (d.get("sexism_analysis") or "").strip()
        rsn = (d.get("reasoning") or "").strip()
        t22 = d.get("task2_2", {}) or {}
        ir = (t22.get("intention_reasoning") or "").strip()
        irony = t22.get("irony_detected", False)
        irc = float(t22.get("irony_confidence", 0.0) or 0.0)
        irony_s = f"Irony detected (conf {irc:.2f})" if irony else "No irony"
        txt = f"{ocr} </s> {desc} </s> INTENTION: {ir} </s> IRONY: {irony_s} </s> {sa} </s> {rsn}"
        ip = t22.get("intention_probabilities", {}) or {}
        feat = np.array([
            float(d.get("task2_1", {}).get("sexist_probability", 0.0) or 0.0),
            float(d.get("task2_1", {}).get("confidence", 0.0) or 0.0),
            float(ip.get("NO", 0.0) or 0.0), float(ip.get("DIRECT", 0.0) or 0.0),
            float(ip.get("JUDGEMENTAL", 0.0) or 0.0), float(bool(irony)), irc,
        ], dtype=np.float32)
        return txt, feat

    for part in ("train", "val", "test"):
        for e in splits[part]:
            e["text22"], e["gfeat22"] = enrich(e)
            e["soft22"] = soft22.get(e["id"])         # None en test
            e["maj22"] = maj22.get(e["id"])
    # --- data augmentation JUDGEMENTAL (Idea 2): paráfrasis del OCR de los memes JUDG
    pp = os.path.join(C.PRE_DIR, "judg_paraphrases.json")
    if os.path.exists(pp) and os.getenv("USE_AUG","0")=="1":
        para = json.load(open(pp))
        extra = []
        for e in list(splits["train"]):
            if e.get("maj22") != "JUDGEMENTAL":
                continue
            for k, pv in enumerate(para.get(e["id"], []) or []):
                pv = str(pv).strip()
                if not pv:
                    continue
                # reconstruir text22 sustituyendo el OCR (primer trozo antes del primer </s>)
                rest = e["text22"].split(" </s> ", 1)
                new_text = pv + (" </s> " + rest[1] if len(rest) > 1 else "")
                ne = dict(e); ne["id"] = f"{e['id']}_p{k}"; ne["text22"] = new_text
                extra.append(ne)
        if extra:
            splits["train"] = splits["train"] + extra
            print(f"  [aug] +{len(extra)} ejemplos JUDGEMENTAL parafraseados (total train={len(splits['train'])})")
    return splits


class DS22(Dataset):
    def __init__(self, ex, tok):
        self.ex, self.tok = ex, tok

    def __len__(self): return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        return dict(id=e["id"], text=e["text22"],
                    gfeat=torch.from_numpy(e["gfeat22"]),
                    emotions=torch.from_numpy(np.asarray(e.get("emotions", np.zeros(C.N_EMOTIONS)), np.float32)),
                    eeg=torch.from_numpy(e["sensors_z"]["EEG"]),
                    soft=torch.tensor(e["soft22"] if e["soft22"] is not None else [-1, -1, -1], dtype=torch.float32))


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j, :n] = x["eeg"]; mask[j, :n] = True
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    gfeat=torch.stack([x["gfeat"] for x in b]), emotions=torch.stack([x["emotions"] for x in b]),
                    eeg=eeg, eeg_mask=mask, soft=torch.stack([x["soft"] for x in b]))
    return f


# ---------------------------------------------------------------- modelo
class HierarchicalHead(nn.Module):
    def __init__(self, fd, p=0.3):
        super().__init__()
        self.bin_head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, 1))
        self.type_head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, 2))

    def forward(self, x):
        lb = self.bin_head(x).squeeze(-1)
        lt = self.type_head(x)
        ps = torch.sigmoid(lb)
        pt = torch.softmax(lt, dim=-1)
        probs = torch.stack([1 - ps, ps * pt[:, 0], ps * pt[:, 1]], dim=-1)
        return probs, lb, lt


class VistaE22(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(C.TEXT_MODEL, torch_dtype=C.AMP_DTYPE,
                                                    attn_implementation=C.best_attn_impl())
        self.text_model.gradient_checkpointing_enable()
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS + 7
        self.head = HierarchicalHead(fd, C.DROPOUT)

    def forward(self, b):
        t = encode_text_mean_pool(self.text_model, b["input_ids"], b["attention_mask"]).float()
        e, _ = self.eeg_pool(b["eeg"].float(), b["eeg_mask"])
        x = torch.cat([t, e, b["emotions"].float(), b["gfeat"].float()], dim=1)
        return self.head(x)


def warm_start(model):
    p = os.path.join(C.CKPT_DIR, "M3_vista_E_best.pt")
    if not os.path.exists(p):
        print("  [warm-start] no hay checkpoint Vista E 2.1 -> init aleatoria"); return
    sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    tm = {k[len("text_model."):]: v for k, v in sd.items() if k.startswith("text_model.")}
    model.text_model.load_state_dict(tm, strict=False)
    ep = {k[len("pools.EEG."):]: v for k, v in sd.items() if k.startswith("pools.EEG.")}
    if ep: model.eeg_pool.load_state_dict(ep, strict=False)
    print(f"  [warm-start] cargado XLM-R ({len(tm)} tensores) + EEG pool ({len(ep)}) desde Vista E 2.1")


# ---------------------------------------------------------------- loss / sampler
def loss_fn(probs, lb, lt, soft, alpha=0.5, gamma=2.0):
    tgt_sex = (1.0 - soft[:, 0]).clamp(0, 1)
    Lb = F.binary_cross_entropy_with_logits(lb, tgt_sex)
    den = tgt_sex.clamp(min=1e-6)
    tgt_type = torch.stack([soft[:, 1] / den, soft[:, 2] / den], dim=-1).clamp(0, 1)
    logp = F.log_softmax(lt, dim=-1); pp = logp.exp()
    af = torch.tensor([1.0, 2.0], device=lt.device)
    fw = (1 - pp).pow(gamma)
    Lt_i = -(tgt_type * af * fw * logp).sum(-1)
    m = (tgt_sex > 0.5).float()
    Lt = (Lt_i * m).sum() / m.sum().clamp(min=1)
    return Lb + alpha * Lt, Lb.item(), Lt.item()


def sampler_for(ex):
    labs = [e["maj22"] for e in ex]
    freq = {x: max(1, labs.count(x)) for x in INT}
    w = torch.tensor([1.0 / math.sqrt(freq[l]) for l in labs])
    return WeightedRandomSampler(w, len(w), replacement=True)


# ---------------------------------------------------------------- eval / inferencia
@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, P, T = [], [], []
    for b in dl:
        bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            probs, _, _ = model(bb)
        ids += b["id"]; P.append(probs.float().cpu().numpy()); T.append(b["soft"].numpy())
    return ids, np.concatenate(P), np.concatenate(T)


def f1macro(probs, soft):
    y = np.argmax(soft, 1); pred = np.argmax(probs, 1)
    return f1_score(y, pred, average="macro", labels=[0, 1, 2])


def thr2d(probs, soft):
    y = np.argmax(soft, 1); best = (-1, 0.30, 0.40)
    grid_j = np.arange(0.10, 0.45, 0.025); grid_d = np.arange(0.25, 0.60, 0.025)
    for tj in grid_j:
        for td in grid_d:
            pred = np.where(probs[:, 2] > tj, 2, np.where(probs[:, 1] > td, 1, 0))
            s = f1_score(y, pred, average="macro", labels=[0, 1, 2])
            if s > best[0]: best = (s, tj, td)
    return best[1], best[2], best[0]


def apply_thr(probs, tj, td):
    return np.where(probs[:, 2] > tj, 2, np.where(probs[:, 1] > td, 1, 0))


# ---------------------------------------------------------------- train
def train():
    C.set_seed(SEED)
    print("[Task 2.2] cargando datos ...")
    splits = load_task22()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    cl = collate(tok)
    ds_tr, ds_va = DS22(splits["train"], tok), DS22(splits["val"], tok)
    dl_tr = DataLoader(ds_tr, batch_size=16, sampler=sampler_for(splits["train"]),
                       collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, collate_fn=cl, num_workers=4)

    model = VistaE22().to(DEV)
    warm_start(model)

    def run_epoch(opt, sched):
        model.train()
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                probs, lb, lt = model(bb)
                loss, _, _ = loss_fn(probs, lb, lt, bb["soft"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    def ev(tag):
        ids, P, T = infer(model, dl_va)
        f1 = f1macro(P, T)
        ce = float(np.mean(-(T * np.log(np.clip(P, 1e-7, 1))).sum(1)))
        print(f"  [{tag}] val F1macro={f1:.4f}  softCE={ce:.4f}")
        return f1

    # Fase 1: XLM-R congelado
    for p in model.text_model.parameters(): p.requires_grad = False
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"[Task 2.2] FASE 1 ({P1_EPOCHS} ép, head+EEG, {sum(p.numel() for p in tr):,} params)")
    opt = torch.optim.AdamW(tr, lr=5e-5, weight_decay=0.01)
    st = max(1, len(dl_tr) * P1_EPOCHS)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    for e in range(1, P1_EPOCHS + 1): run_epoch(opt, sch); ev(f"F1 {e}/{P1_EPOCHS}")
    del opt, sch; torch.cuda.empty_cache()

    # Fase 2: XLM-R descongelado, LRs diferenciados
    for p in model.text_model.parameters(): p.requires_grad = True
    low, high = [], []
    for n, p in model.text_model.named_parameters():
        ln = None
        if "encoder.layer." in n: ln = int(n.split("encoder.layer.")[1].split(".")[0])
        (low if (("embeddings" in n) or (ln is not None and ln <= 6)) else high).append(p)
    head = [p for n, p in model.named_parameters() if not n.startswith("text_model.")]
    opt = torch.optim.AdamW([{"params": low, "lr": 1e-5}, {"params": high, "lr": 3e-5},
                             {"params": head, "lr": 1e-4}], weight_decay=0.01)
    st = max(1, len(dl_tr) * P2_EPOCHS)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    print(f"[Task 2.2] FASE 2 ({P2_EPOCHS} ép, full fine-tune)")
    best_f1, best_sd, pat = -1, None, 0
    for e in range(1, P2_EPOCHS + 1):
        run_epoch(opt, sch); f1 = ev(f"F2 {e}/{P2_EPOCHS}")
        if f1 > best_f1:
            best_f1, best_sd, pat = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
            print(f"     mejor F1macro={f1:.4f} -> checkpoint")
        else:
            pat += 1
            if pat >= PATIENCE: print("     early stopping"); break
    if best_sd: model.load_state_dict(best_sd)
    torch.save(dict(model_state_dict=best_sd, val_f1macro=best_f1), os.path.join(C.CKPT_DIR, "vista_e_task22_best.pt"))
    print(f"[Task 2.2] checkpoint vista_e_task22_best.pt  (val F1macro={best_f1:.4f})")

    # ---- threshold 2D sobre validación
    vids, vP, vT = infer(model, dl_va)
    tj, td, f1opt = thr2d(vP, vT)
    print(f"[Task 2.2] thresholds 2D: t_JUDG={tj:.3f}  t_DIRECT={td:.3f}  (val F1macro={f1opt:.4f})")

    # ---- comparativa en validación
    yv = np.argmax(vT, 1)
    g_val = np.array([_gemini_intprobs(i) for i in vids])
    print("=== Task 2.2 — VALIDACIÓN ===")
    print(f"  Gemini crudo (argmax)    F1macro={f1_score(yv, np.argmax(g_val,1), average='macro', labels=[0,1,2]):.4f}")
    print(f"  Vista E (argmax)         F1macro={f1_score(yv, np.argmax(vP,1), average='macro', labels=[0,1,2]):.4f}")
    print(f"  Vista E (thr 2D)         F1macro={f1opt:.4f}")
    blend = 0.6 * vP + 0.4 * g_val
    print(f"  0.6*VistaE + 0.4*Gemini  F1macro={f1_score(yv, np.argmax(blend,1), average='macro', labels=[0,1,2]):.4f}")

    # ---- inferencia test
    ds_te = DS22(splits["test"], tok)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, collate_fn=cl, num_workers=4)
    tids, tP, _ = infer(model, dl_te)
    g_test = np.array([_gemini_intprobs(i) for i in tids])
    return splits, tids, tP, g_test, tj, td


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


# ---------------------------------------------------------------- submissions + zip
def write_runs(tids, tP, g_test, tj, td):
    blend = 0.6 * tP + 0.4 * g_test
    runs = {  # (run_id, kind) -> (probs3, hard_pred_idx or None)
        ("1", "soft"): (g_test, None), ("1", "hard"): (g_test, np.argmax(g_test, 1)),
        ("2", "soft"): (blend, None), ("2", "hard"): (blend, np.argmax(blend, 1)),
        ("3", "soft"): (tP, None), ("3", "hard"): (tP, apply_thr(tP, tj, td)),
    }
    OUT, TC = C.OUT_DIR, "EXIST2025"
    for (n, kind), (probs, hard) in runs.items():
        out = []
        for i, mid in enumerate(tids):
            if kind == "hard":
                out.append({"test_case": TC, "id": str(mid), "value": INT[int(hard[i])]})
            else:
                p = probs[i]; p = p / max(p.sum(), 1e-9)
                out.append({"test_case": TC, "id": str(mid),
                            "value": {"NO": float(p[0]), "DIRECT": float(p[1]), "JUDGEMENTAL": float(p[2])}})
        json.dump(out, open(os.path.join(OUT, f"task2_2_{kind}_{C.TEAM_NAME}_{n}"), "w"), indent=2)
        print(f"  escrito task2_2_{kind}_{C.TEAM_NAME}_{n}")

    # validar formato + re-zip TODO
    fns = sorted(f for f in os.listdir(OUT) if f.startswith("task2_"))
    test_ids = set(str(i) for i in tids)
    for fn in fns:
        data = json.load(open(os.path.join(OUT, fn))); ids = set()
        for it in data:
            assert set(it.keys()) == {"test_case", "id", "value"} and it["test_case"] == TC
            ids.add(it["id"])
            v = it["value"]
            if "task2_1" in fn:
                assert (set(v) == {"YES", "NO"} and abs(sum(v.values()) - 1) < 1e-4) if "soft" in fn else v in {"YES", "NO"}
            elif "task2_2" in fn:
                assert (set(v) == set(INT) and abs(sum(v.values()) - 1) < 1e-3) if "soft" in fn else v in INT
            elif "task2_3" in fn:
                assert (set(v) == {"NO", "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
                                   "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"}) if "soft" in fn \
                       else (isinstance(v, list) and len(v) >= 1)
        assert ids == test_ids, fn
    zp = os.path.join(C.ROOT, f"exist2026_{C.TEAM_NAME}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in fns: zf.write(os.path.join(OUT, fn), arcname=os.path.join(f"exist2026_{C.TEAM_NAME}", fn))
    print(f"\nZIP final: {zp}  ({os.path.getsize(zp)/1024:.1f} KB, {len(fns)} ficheros)")
    for fn in fns: print("  -", fn)


if __name__ == "__main__":
    C.configure_gpu()
    splits, tids, tP, g_test, tj, td = train()
    write_runs(tids, tP, g_test, tj, td)
    print("\nMapeo Task 2.2:  run1=Gemini crudo · run2=0.6·VistaE+0.4·Gemini · run3=VistaE adaptada (thr 2D)")
