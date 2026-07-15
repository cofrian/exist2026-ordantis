"""Task 2.3 (categorías de sexismo, multi-label de 5 clases) — Vista E adaptada.

Plan ajustado:
  FASE 0  -> entrenar Vista E base (5 sigmoides indep., Asymmetric Loss, warm-start
             desde Vista E 2.1, texto enriquecido con category_reasoning de Gemini,
             soft labels multi-label) y diagnosticar contra Gemini crudo.
  FASE 1  -> Platt scaling por clase + thresholds por clase con protección a minorías.
  FASE 2  -> (opcional, sólo si Vista E supera a Gemini) paráfrasis de minorías +
             auxiliary task de desacuerdo.  No implementada por defecto aquí.

NO usa category_probabilities de Gemini como features ni en blends (lección Task 2.2).
"""
import json, os, math, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

import config as C
import data as D
from models import SetAttentionPool, encode_text_mean_pool

CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
        "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]
NC = len(CATS)
DEV = C.DEVICE
SEED = 999
P1_EPOCHS, P2_EPOCHS, PATIENCE = 3, 12, 4
MAX_TOK = 320
SUBM_CATS = {"NO"} | set(CATS)   # vocabulario soft de la submission
N_DIS = 11                       # nº de features de desacuerdo entre anotadores (auxiliary task)
USE_AUX = os.environ.get("TASK23_AUX", "1") == "1"


# ---------------------------------------------------------------- datos
def _ann_cats(entry):
    """Categorías marcadas por un anotador; None si el voto es UNKNOWN (no cuenta)."""
    if isinstance(entry, list):
        return [x for x in entry if x in CATS]
    if entry in ("-", "NO", "", None):
        return []
    if entry == "UNKNOWN":
        return None
    return []


def _entropy(probs):
    p = np.asarray([x for x in probs if x > 0], float)
    return float(-(p * np.log(p)).sum()) if len(p) > 1 else 0.0


def disagreement_features(m):
    """11 features de (des)acuerdo entre los anotadores, derivadas de las 3 subtareas. Solo train."""
    n = max(1, int(m.get("number_annotators", 6)))
    t1 = m.get("labels_task2_1", []); t2 = m.get("labels_task2_2", []); t3 = m.get("labels_task2_3", [])
    feats = []
    # entropía binaria task 2.1
    y = sum(1 for v in t1 if v == "YES") / n
    feats.append(_entropy([y, 1 - y]))
    # entropía task 2.2 (NO/DIRECT/JUDGEMENTAL; "-" -> NO)
    c2 = [sum(1 for v in t2 if (("NO" if v == "-" else v) == k)) for k in ("NO", "DIRECT", "JUDGEMENTAL")]
    feats.append(_entropy([x / n for x in c2]))
    # jaccard medio / std entre pares de anotadores en task 2.3
    sets = [set(_ann_cats(v) or []) for v in t3]
    ov = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            ov.append(1.0 if not u else len(sets[i] & sets[j]) / len(u))
    feats.append(float(np.mean(ov)) if ov else 1.0)
    feats.append(float(np.std(ov)) if ov else 0.0)
    # consenso por categoría (5)
    for c in CATS:
        feats.append(sum(1 for s in sets if c in s) / n)
    # diversidad de categorías votadas y % anotadores que no marcan nada
    allc = set().union(*sets) if sets else set()
    feats.append(len(allc) / len(CATS))
    feats.append(sum(1 for s in sets if not s) / n)
    return np.array(feats, np.float32)  # 1+1+2+5+1+1 = 11


def load_task23():
    splits = D.load_split()
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    train_raw = json.load(open(C.TRAIN_JSON, encoding="utf-8"))

    soft23, dis23 = {}, {}
    for m in train_raw.values():
        votes = [_ann_cats(v) for v in m["labels_task2_3"]]
        votes = [v for v in votes if v is not None]
        n = max(1, len(votes))
        mid = str(m["id_EXIST"])
        soft23[mid] = [sum(1 for v in votes if c in v) / n for c in CATS]
        dis23[mid] = disagreement_features(m)

    def gd(mid):
        v = g.get(str(mid))
        return v if isinstance(v, dict) else None

    def enrich(e):
        d = gd(e["id"]); ocr = e["text"]
        if not d:
            return ocr
        desc = (d.get("description") or "").strip()
        sa = (d.get("sexism_analysis") or "").strip()
        rsn = (d.get("reasoning") or "").strip()
        cr = ((d.get("task2_3", {}) or {}).get("category_reasoning") or "").strip()
        return f"{ocr} </s> {desc} </s> {sa} </s> {rsn} </s> CATEGORIES: {cr}"

    for part in ("train", "val", "test"):
        for e in splits[part]:
            e["text23"] = enrich(e)
            e["soft23"] = soft23.get(e["id"])          # None en test
            e["dis23"] = dis23.get(e["id"])            # None en test

    # data augmentation de minorías (paráfrasis del análisis): opcional, fase 2
    pp = os.path.join(C.PRE_DIR, "task23_paraphrases.json")
    if os.path.exists(pp):
        para = json.load(open(pp)); extra = []
        for e in list(splits["train"]):
            s = e["soft23"] or [0] * NC
            is_minor = (s[3] >= 0.5) or (s[4] >= 0.5)   # SEXUAL-VIOLENCE / MISOGYNY-NSV
            if not is_minor:
                continue
            for k, pv in enumerate(para.get(e["id"], []) or []):
                pv = (pv.get("category_reasoning") if isinstance(pv, dict) else str(pv)).strip()
                if not pv:
                    continue
                rest = e["text23"].split(" </s> CATEGORIES: ", 1)
                ne = dict(e); ne["id"] = f"{e['id']}_p{k}"
                ne["text23"] = rest[0] + " </s> CATEGORIES: " + pv
                extra.append(ne)
        if extra:
            splits["train"] += extra
            print(f"  [aug] +{len(extra)} ejemplos minoritarios parafraseados (total={len(splits['train'])})")
    return splits


class DS23(Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]
        return dict(id=e["id"], text=e["text23"],
                    emotions=torch.from_numpy(np.asarray(e.get("emotions", np.zeros(C.N_EMOTIONS)), np.float32)),
                    eeg=torch.from_numpy(e["sensors_z"]["EEG"]),
                    soft=torch.tensor(e["soft23"] if e["soft23"] is not None else [-1.0] * NC, dtype=torch.float32),
                    dis=torch.from_numpy(e["dis23"] if e["dis23"] is not None else np.full(N_DIS, -1.0, np.float32)))


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j, :n] = x["eeg"]; mask[j, :n] = True
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    emotions=torch.stack([x["emotions"] for x in b]),
                    eeg=eeg, eeg_mask=mask, soft=torch.stack([x["soft"] for x in b]),
                    dis=torch.stack([x["dis"] for x in b]))
    return f


# ---------------------------------------------------------------- modelo
class MultiLabelHead(nn.Module):
    def __init__(self, fd, nc=NC, p=0.3):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(256, 128), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(128, nc))
    def forward(self, x):
        logits = self.head(x)
        return torch.sigmoid(logits), logits


class VistaE23(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(C.TEXT_MODEL, torch_dtype=C.AMP_DTYPE,
                                                    attn_implementation=C.best_attn_impl())
        self.text_model.gradient_checkpointing_enable()
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS
        self.head = MultiLabelHead(fd, NC, C.DROPOUT)
        self.aux = nn.Sequential(nn.Linear(fd, 128), nn.GELU(), nn.Linear(128, N_DIS)) if USE_AUX else None
    def forward(self, b):
        t = encode_text_mean_pool(self.text_model, b["input_ids"], b["attention_mask"]).float()
        e, _ = self.eeg_pool(b["eeg"].float(), b["eeg_mask"])
        x = torch.cat([t, e, b["emotions"].float()], dim=1)
        probs, logits = self.head(x)
        dis_pred = self.aux(x) if self.aux is not None else None
        return probs, logits, dis_pred


def warm_start(model):
    for cand in ("vista_e_task22_best.pt", "M3_vista_E_best.pt"):
        p = os.path.join(C.CKPT_DIR, cand)
        if os.path.exists(p):
            sd = torch.load(p, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            tm = {k[len("text_model."):]: v for k, v in sd.items() if k.startswith("text_model.")}
            model.text_model.load_state_dict(tm, strict=False)
            ep = {k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("eeg_pool.") or k.startswith("pools.EEG.")}
            if ep: model.eeg_pool.load_state_dict(ep, strict=False)
            print(f"  [warm-start] {cand}: XLM-R {len(tm)} tensores, EEG pool {len(ep)}")
            return
    print("  [warm-start] sin checkpoint previo -> init aleatoria")


# ---------------------------------------------------------------- loss
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__(); self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits); xs_neg = 1 - xs_pos
        if self.clip > 0: xs_neg = (xs_neg + self.clip).clamp(max=1)
        los = targets * torch.log(xs_pos.clamp(min=self.eps)) + (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        pt = xs_pos * targets + xs_neg * (1 - targets)
        g = self.gp * targets + self.gn * (1 - targets)
        return -(los * torch.pow(1 - pt, g)).mean()


def sampler_for(ex):
    """Sobremuestrea memes que contienen alguna categoría rara (SV / MISOGYNY-NSV)."""
    w = []
    for e in ex:
        s = e["soft23"] or [0] * NC
        rare = (s[3] >= 0.5) or (s[4] >= 0.5)
        med = (s[0] >= 0.5) or (s[1] >= 0.5) or (s[2] >= 0.5)
        w.append(3.0 if rare else (1.0 if med else 0.5))
    return WeightedRandomSampler(torch.tensor(w), len(w), replacement=True)


# ---------------------------------------------------------------- eval / inferencia
@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, P, T, Dz = [], [], [], []
    for b in dl:
        bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            probs, _, dis = model(bb)
        ids += b["id"]; P.append(probs.float().cpu().numpy()); T.append(b["soft"].numpy())
        Dz.append(dis.float().cpu().numpy() if dis is not None else np.zeros((len(b["id"]), N_DIS), np.float32))
    return ids, np.concatenate(P), np.concatenate(T), np.concatenate(Dz)


def macro_f1(hard, y_hard):
    return f1_score(y_hard, hard, average="macro", zero_division=0)


def icmsoft_proxy(probs, soft):
    """Proxy de ICMSoft: BCE soft media (menor = mejor; lo reportamos negado)."""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    bce = -(soft * np.log(p) + (1 - soft) * np.log(1 - p)).mean()
    return -bce


# ---- Fase 1: Platt por clase + thresholds protegidos
def fit_platt(val_probs, val_y):
    out = []
    for i in range(NC):
        p = np.clip(val_probs[:, i], 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        if len(np.unique(val_y[:, i])) < 2:
            out.append((1.0, 0.0)); continue
        lr = LogisticRegression(C=1.0); lr.fit(z, val_y[:, i])
        out.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
    return out


def apply_platt(probs, params):
    cal = np.zeros_like(probs)
    for i, (a, b) in enumerate(params):
        p = np.clip(probs[:, i], 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p))
        cal[:, i] = 1 / (1 + np.exp(-(a * z + b)))
    return cal


def find_thresholds(val_probs, val_y, class_freqs):
    ths = []
    for i in range(NC):
        freq = class_freqs[i]
        pw = max(1.0, 0.20 / max(freq, 0.01))
        best = (-1e9, 0.5, 0.0)
        for t in np.arange(0.05, 0.80, 0.02):
            pred = (val_probs[:, i] > t).astype(int); tr = val_y[:, i]
            tp = ((pred == 1) & (tr == 1)).sum(); fp = ((pred == 1) & (tr == 0)).sum(); fn = ((pred == 0) & (tr == 1)).sum()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            score = f1 + pw * 0.10 * (rec - 0.5)
            if score > best[0]: best = (score, t, f1)
        ths.append(best[1])
        print(f"  thr {CATS[i]:<32} t={best[1]:.3f}  F1={best[2]:.4f}  peso={pw:.2f}")
    return np.array(ths)


# ---------------------------------------------------------------- train
def train():
    C.set_seed(SEED)
    print("[Task 2.3] cargando datos ...")
    splits = load_task23()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL); cl = collate(tok)
    dl_tr = DataLoader(DS23(splits["train"]), batch_size=16, sampler=sampler_for(splits["train"]),
                       collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS23(splits["val"]), batch_size=64, shuffle=False, collate_fn=cl, num_workers=4)

    # frecuencias de clase en train (proporción de memes con prob mayoritaria de la cat)
    n_tr = len(splits["train"])
    class_freqs = [sum(1 for e in splits["train"] if (e["soft23"] or [0]*NC)[i] >= 0.5) / max(n_tr, 1) for i in range(NC)]
    print("  freqs:", {CATS[i]: round(class_freqs[i], 3) for i in range(NC)})

    model = VistaE23().to(DEV); warm_start(model)
    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    def run_epoch(opt, sch):
        model.train()
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                _, logits, dis = model(bb); loss = asl(logits, bb["soft"])
                if dis is not None:
                    m = (bb["dis"][:, 0] >= 0).float().unsqueeze(1)
                    if m.sum() > 0:
                        loss = loss + 0.1 * (((dis - bb["dis"]) ** 2) * m).sum() / (m.sum() * N_DIS)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

    def ev(tag):
        ids, P, T, _ = infer(model, dl_va)
        yh = (T >= 0.5).astype(int)
        hard = (P > 0.5).astype(int)
        f1 = macro_f1(hard, yh); soft = icmsoft_proxy(P, T)
        print(f"  [{tag}] valF1macro(t.5)={f1:.4f}  ICMSoftProxy={soft:.4f}")
        return f1

    # Fase 1: XLM-R congelado
    for p in model.text_model.parameters(): p.requires_grad = False
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"[Task 2.3] FASE 1 ({P1_EPOCHS} ép, head+EEG, {sum(p.numel() for p in tr):,} params)")
    opt = torch.optim.AdamW(tr, lr=5e-5, weight_decay=0.01)
    st = max(1, len(dl_tr) * P1_EPOCHS); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    for e in range(1, P1_EPOCHS + 1): run_epoch(opt, sch); ev(f"F1 {e}/{P1_EPOCHS}")
    del opt, sch; torch.cuda.empty_cache()

    # Fase 2: XLM-R descongelado
    for p in model.text_model.parameters(): p.requires_grad = True
    low, high = [], []
    for n, p in model.text_model.named_parameters():
        ln = int(n.split("encoder.layer.")[1].split(".")[0]) if "encoder.layer." in n else None
        (low if (("embeddings" in n) or (ln is not None and ln <= 6)) else high).append(p)
    head = [p for n, p in model.named_parameters() if not n.startswith("text_model.")]
    opt = torch.optim.AdamW([{"params": low, "lr": 1e-5}, {"params": high, "lr": 3e-5},
                             {"params": head, "lr": 1e-4}], weight_decay=0.01)
    st = max(1, len(dl_tr) * P2_EPOCHS); sch = get_linear_schedule_with_warmup(opt, int(0.1 * st), st)
    print(f"[Task 2.3] FASE 2 ({P2_EPOCHS} ép, full fine-tune)")
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
    torch.save(dict(model_state_dict=best_sd, val_f1macro=best_f1), os.path.join(C.CKPT_DIR, "vista_e_task23_best.pt"))
    print(f"[Task 2.3] checkpoint vista_e_task23_best.pt  (val F1macro t.5={best_f1:.4f})")

    # ---- FASE 1: Platt + thresholds protegidos sobre validación
    vids, vP, vT, vD = infer(model, dl_va); vY = (vT >= 0.5).astype(int)
    platt = fit_platt(vP, vY); vPc = apply_platt(vP, platt)
    print("[Task 2.3] thresholds protegidos:")
    ths = find_thresholds(vPc, vY, class_freqs)
    vhard = (vPc > ths[None, :]).astype(int)

    # ---- diagnóstico vs Gemini crudo
    gV = np.array([_gemini_catprobs(i) for i in vids])
    ghard = (gV > 0.5).astype(int)
    print("=== Task 2.3 — VALIDACIÓN ===")
    print(f"  Gemini crudo            F1macro={macro_f1(ghard, vY):.4f}  ICMSoftProxy={icmsoft_proxy(gV, vT):.4f}")
    print(f"  Vista E (t .5)          F1macro={macro_f1((vP>.5).astype(int), vY):.4f}  ICMSoftProxy={icmsoft_proxy(vP, vT):.4f}")
    print(f"  Vista E (Platt+thr)     F1macro={macro_f1(vhard, vY):.4f}  ICMSoftProxy={icmsoft_proxy(vPc, vT):.4f}")
    for i in range(NC):
        print(f"    {CATS[i]:<32} F1={f1_score(vY[:,i], vhard[:,i], zero_division=0):.4f}")
    decision = "FULL" if macro_f1(vhard, vY) > macro_f1(ghard, vY) + 0.01 else \
               ("REFINE_ONLY" if macro_f1(vhard, vY) > macro_f1(ghard, vY) - 0.01 else "DEBUG")
    print(f"  -> DECISIÓN FASE 0: {decision}")

    # ---- inferencia test
    dl_te = DataLoader(DS23(splits["test"]), batch_size=64, shuffle=False, collate_fn=cl, num_workers=4)
    tids, tP, _, tD = infer(model, dl_te)
    tPc = apply_platt(tP, platt)
    gT = np.array([_gemini_catprobs(i) for i in tids])
    # ajuste de threshold por desacuerdo predicho (auxiliary task): memes ambiguos -> más conservador
    dmag = np.clip(np.abs(tD).mean(axis=1), 0, 1) if USE_AUX else np.zeros(len(tids))
    return tids, tP, tPc, ths, gT, dmag


_G = None
def _gemini_catprobs(mid):
    """Probs de categorías de Gemini SOLO para comparación/submission backup (no como feature)."""
    global _G
    if _G is None: _G = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    v = _G.get(str(mid))
    if isinstance(v, dict):
        cp = (v.get("task2_3", {}) or {}).get("category_probabilities", {}) or {}
        return np.array([float(cp.get(c, 0.0) or 0.0) for c in CATS], np.float32)
    return np.zeros(NC, np.float32)


# ---------------------------------------------------------------- submissions
def _hard_value(probs_row, thr_row):
    cats = [CATS[i] for i in range(NC) if probs_row[i] > thr_row[i]]
    if not cats:  # nunca vacío: el formato exige >=1 -> coger el argmax
        cats = [CATS[int(np.argmax(probs_row))]]
    return cats


def _soft_value(probs_row):
    d = {c: float(probs_row[i]) for i, c in enumerate(CATS)}
    d["NO"] = float(max(0.0, 1.0 - max(probs_row)))
    return d


def write_runs(tids, tP, tPc, ths, gT, dmag):
    g_ths = np.full(NC, 0.5)
    # run3: threshold base protegido + ajuste por desacuerdo predicho (hasta +0.10), por meme
    thr3 = ths[None, :] + np.clip(dmag, 0, 1)[:, None] * 0.10
    runs = {
        ("1", "soft"): (gT, None), ("1", "hard"): (gT, g_ths),                  # Gemini crudo (backup)
        ("2", "soft"): (tPc, None), ("2", "hard"): (tPc, ths),                   # Vista E Platt + thr protegido
        ("3", "soft"): (tPc, None), ("3", "hard"): (tPc, thr3),                  # Vista E + aux (thr adaptativo)
    }
    OUT, TC = C.OUT_DIR, "EXIST2025"
    for (n, kind), (probs, thr) in runs.items():
        out = []
        for i, mid in enumerate(tids):
            tr = thr[i] if (thr is not None and thr.ndim == 2) else thr
            v = _hard_value(probs[i], tr) if kind == "hard" else _soft_value(probs[i])
            out.append({"test_case": TC, "id": str(mid), "value": v})
        json.dump(out, open(os.path.join(OUT, f"task2_3_{kind}_{C.TEAM_NAME}_{n}"), "w"), indent=2)
        print(f"  escrito task2_3_{kind}_{C.TEAM_NAME}_{n}")

    fns = sorted(f for f in os.listdir(OUT) if f.startswith("task2_"))
    test_ids = set(str(i) for i in tids)
    for fn in fns:
        if "task2_3" not in fn: continue
        data = json.load(open(os.path.join(OUT, fn))); ids = set()
        for it in data:
            assert set(it.keys()) == {"test_case", "id", "value"} and it["test_case"] == TC
            ids.add(it["id"]); v = it["value"]
            if "soft" in fn:
                assert set(v) == SUBM_CATS and all(0 <= x <= 1 for x in v.values()), fn
            else:
                assert isinstance(v, list) and len(v) >= 1 and all(x in CATS for x in v), fn
        assert ids == test_ids, fn
    zp = os.path.join(C.ROOT, f"exist2026_{C.TEAM_NAME}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in fns: zf.write(os.path.join(OUT, fn), arcname=os.path.join(f"exist2026_{C.TEAM_NAME}", fn))
    print(f"\nZIP final: {zp}  ({os.path.getsize(zp)/1024:.1f} KB, {len(fns)} ficheros)")


if __name__ == "__main__":
    C.configure_gpu()
    tids, tP, tPc, ths, gT, dmag = train()
    write_runs(tids, tP, tPc, ths, gT, dmag)
    print("\nMapeo Task 2.3:  run1=Gemini crudo · run2=VistaE Platt+thr protegido · run3=VistaE+aux (thr adaptativo)")
