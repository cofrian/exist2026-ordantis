"""EXPERIMENTO B: Task 2.2 con XLM-R-Longformer-base-4096 (multilingüe, 4096 tokens
de contexto). Texto enriquecido completo (con reasoning). max_length=1100 (cubre el
máximo de nuestros datos 1041). NO toca la submission actual: guarda checkpoint en
_alt/ y predicciones aparte. NO warm-start (el Longformer no es compatible)."""
import json, os, tempfile, math
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
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

ALT = os.path.join(C.OUT_DIR, "_alt"); os.makedirs(ALT, exist_ok=True)
INT = ["NO", "DIRECT", "JUDGEMENTAL"]; HIER = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
TC = "EXIST2025"; EPS = 1e-7; SEED = 999
LONG_MODEL = "markussagen/xlm-roberta-longformer-base-4096"
MAX_TOK = 1100   # ← cubre nuestro máximo 1041
P1, P2, PAT = 2, 8, 3   # menos épocas porque cada una es ~2.5x más cara
DEV = C.DEVICE
CKPT = os.path.join(ALT, "vista_e_task22_longformer_best.pt")


def load_t22():
    splits = D.load_split()
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    raw = json.load(open(C.TRAIN_JSON, encoding="utf-8"))
    soft, maj = {}, {}
    for m in raw.values():
        votes = [("NO" if v == "-" else v) for v in m["labels_task2_2"] if v != "UNKNOWN"]
        c = {x: votes.count(x) for x in INT}; n = sum(c.values())
        soft[str(m["id_EXIST"])] = ([c[x]/n for x in INT] if n else [1.0,0.0,0.0])
        maj[str(m["id_EXIST"])] = (max(c, key=c.get) if n else "NO")
    def gd(mid):
        v = g.get(str(mid)); return v if isinstance(v, dict) else None
    def enrich(e):
        d = gd(e["id"]); ocr = e["text"]
        if not d: return ocr, np.zeros(7, np.float32)
        desc = (d.get("description") or "").strip()
        sa = (d.get("sexism_analysis") or "").strip()
        rsn = (d.get("reasoning") or "").strip()
        t22 = d.get("task2_2", {}) or {}
        ir = (t22.get("intention_reasoning") or "").strip()
        irony = t22.get("irony_detected", False)
        irc = float(t22.get("irony_confidence", 0.0) or 0.0)
        irony_s = f"Irony detected (conf {irc:.2f})" if irony else "No irony"
        # texto COMPLETO (con reasoning) — el longformer no trunca
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
            e["t22_text"], e["t22_feat"] = enrich(e)
            e["t22_soft"] = soft.get(e["id"]); e["t22_maj"] = maj.get(e["id"])
    return splits


class DS22(Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]
        return dict(id=e["id"], text=e["t22_text"], feat=torch.from_numpy(e["t22_feat"]),
                    emotions=torch.from_numpy(np.asarray(e.get("emotions", np.zeros(C.N_EMOTIONS)), np.float32)),
                    eeg=torch.from_numpy(e["sensors_z"]["EEG"]),
                    soft=torch.tensor(e["t22_soft"] if e["t22_soft"] is not None else [-1,-1,-1], dtype=torch.float32))


def collate(tok):
    def f(b):
        enc = tok([x["text"] for x in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        S = max(x["eeg"].shape[0] for x in b)
        eeg = torch.zeros(len(b), S, C.N_EEG); mask = torch.zeros(len(b), S, dtype=torch.bool)
        for j, x in enumerate(b):
            n = x["eeg"].shape[0]; eeg[j,:n] = x["eeg"]; mask[j,:n] = True
        # global attention: ponemos el primer token global (estilo CLS)
        gam = torch.zeros_like(enc["input_ids"]); gam[:, 0] = 1
        return dict(id=[x["id"] for x in b], input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    global_attention_mask=gam,
                    feat=torch.stack([x["feat"] for x in b]), emotions=torch.stack([x["emotions"] for x in b]),
                    eeg=eeg, eeg_mask=mask, soft=torch.stack([x["soft"] for x in b]))
    return f


class HierHead(nn.Module):
    def __init__(self, fd, p=0.3):
        super().__init__()
        self.bin_head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, 1))
        self.type_head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, 2))
    def forward(self, x):
        lb = self.bin_head(x).squeeze(-1); lt = self.type_head(x)
        ps = torch.sigmoid(lb); pt = torch.softmax(lt, dim=-1)
        return torch.stack([1-ps, ps*pt[:,0], ps*pt[:,1]], dim=-1), lb, lt


class VistaELong(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(LONG_MODEL, torch_dtype=C.AMP_DTYPE)
        try: self.text_model.gradient_checkpointing_enable()
        except Exception: pass
        self.eeg_pool = SetAttentionPool(C.N_EEG, 256)
        fd = self.text_model.config.hidden_size + 256 + C.N_EMOTIONS + 7
        self.head = HierHead(fd, C.DROPOUT)

    def forward(self, b):
        out = self.text_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                              global_attention_mask=b.get("global_attention_mask"))
        last = out.last_hidden_state            # (B, L, H)
        m = b["attention_mask"].unsqueeze(-1).to(last.dtype)
        t = ((last * m).sum(1) / m.sum(1).clamp(min=1e-9)).float()       # mean-pool
        e, _ = self.eeg_pool(b["eeg"].float(), b["eeg_mask"])
        x = torch.cat([t, e, b["emotions"].float(), b["feat"].float()], dim=1)
        return self.head(x)


def loss_fn(probs, lb, lt, soft, alpha=0.5, gamma=2.0):
    tgt = (1.0 - soft[:,0]).clamp(0,1)
    Lb = F.binary_cross_entropy_with_logits(lb, tgt)
    den = tgt.clamp(min=1e-6)
    tt = torch.stack([soft[:,1]/den, soft[:,2]/den], dim=-1).clamp(0,1)
    logp = F.log_softmax(lt, dim=-1); pp = logp.exp()
    af = torch.tensor([1.0, 2.0], device=lt.device); fw = (1-pp).pow(gamma)
    Lti = -(tt*af*fw*logp).sum(-1); m = (tgt>0.5).float()
    Lt = (Lti*m).sum()/m.sum().clamp(min=1)
    return Lb + alpha*Lt


def sampler_for(ex):
    labs = [e["t22_maj"] for e in ex]
    freq = {x: max(1, labs.count(x)) for x in INT}
    w = torch.tensor([1.0/math.sqrt(freq[l]) for l in labs])
    return WeightedRandomSampler(w, len(w), replacement=True)


@torch.no_grad()
def infer(model, dl):
    model.eval(); ids, P, T = [], [], []
    for b in dl:
        bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k,v in b.items()}
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            probs, _, _ = model(bb)
        ids += b["id"]; P.append(probs.float().cpu().numpy()); T.append(b["soft"].numpy())
    return ids, np.concatenate(P), np.concatenate(T)


def dec(p, tj, td): return np.where(p[:,2]>tj, 2, np.where(p[:,1]>td, 1, 0))
def wf1(y, pr): f = f1_score(y, pr, average=None, labels=[0,1,2]); return (f[0]+f[1]+2*f[2])/4


def icm_hard(ids, y, predidx):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td,"p"), os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":INT[int(predidx[k])]} for k,i in enumerate(ids)], open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":INT[int(y[k])]} for k,i in enumerate(ids)], open(gf,"w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICM"], **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,PyEvALLUtils.PARAM_HIERARCHY:HIER,PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        return rep.report["metrics"]["ICM"]["results"]["average_per_test_case"]


def icmsoft(ids, gold_soft, probs):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td,"p"), os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":{INT[c]:float(probs[k,c]/max(probs[k].sum(),EPS)) for c in range(3)}} for k,i in enumerate(ids)], open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":{INT[c]:float(gold_soft[k,c]) for c in range(3)}} for k,i in enumerate(ids)], open(gf,"w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICMSoft"], **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,PyEvALLUtils.PARAM_HIERARCHY:HIER,PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        return rep.report["metrics"]["ICMSoft"]["results"]["average_per_test_case"]


def main():
    C.set_seed(SEED); C.configure_gpu()
    print(f"[EXP-B Longformer] cargando {LONG_MODEL} ... MAX_TOK={MAX_TOK}", flush=True)
    splits = load_t22()
    tok = AutoTokenizer.from_pretrained(LONG_MODEL); cl = collate(tok)
    bs = 4   # batch reducido por longitud
    dl_tr = DataLoader(DS22(splits["train"]), batch_size=bs, sampler=sampler_for(splits["train"]),
                       collate_fn=cl, num_workers=4, pin_memory=True)
    dl_va = DataLoader(DS22(splits["val"]), batch_size=8, shuffle=False, collate_fn=cl, num_workers=4)
    model = VistaELong().to(DEV)

    def run_epoch(opt, sch):
        model.train()
        for b in dl_tr:
            bb = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k,v in b.items()}
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                probs, lb, lt = model(bb); loss = loss_fn(probs, lb, lt, bb["soft"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)

    def ev(tag):
        ids, P, T = infer(model, dl_va); y = np.argmax(T, 1)
        f1 = f1_score(y, np.argmax(P,1), average="macro", labels=[0,1,2])
        print(f"  [{tag}] val F1macro={f1:.4f}", flush=True); return f1

    # Fase 1 congelada
    for p in model.text_model.parameters(): p.requires_grad = False
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"FASE 1 ({P1} ép, head+EEG, {sum(p.numel() for p in tr):,} params)", flush=True)
    opt = torch.optim.AdamW(tr, lr=5e-5, weight_decay=0.01)
    st = max(1, len(dl_tr)*P1); sch = get_linear_schedule_with_warmup(opt, int(0.1*st), st)
    for e in range(1, P1+1): run_epoch(opt, sch); ev(f"F1 {e}/{P1}")
    del opt, sch; torch.cuda.empty_cache()

    # Fase 2
    for p in model.text_model.parameters(): p.requires_grad = True
    low, high = [], []
    for n, p in model.text_model.named_parameters():
        ln = int(n.split("encoder.layer.")[1].split(".")[0]) if "encoder.layer." in n else None
        (low if (("embeddings" in n) or (ln is not None and ln <= 6)) else high).append(p)
    headp = [p for n,p in model.named_parameters() if not n.startswith("text_model.")]
    opt = torch.optim.AdamW([{"params":low,"lr":1e-5},{"params":high,"lr":3e-5},{"params":headp,"lr":1e-4}], weight_decay=0.01)
    st = max(1, len(dl_tr)*P2); sch = get_linear_schedule_with_warmup(opt, int(0.1*st), st)
    print(f"FASE 2 ({P2} ép, full fine-tune)", flush=True)
    best, bsd, pat = -1, None, 0
    for e in range(1, P2+1):
        run_epoch(opt, sch); f1 = ev(f"F2 {e}/{P2}")
        if f1 > best: best, bsd, pat = f1, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0; print(f"     mejor -> ckpt ({f1:.4f})", flush=True)
        else:
            pat += 1
            if pat >= PAT: print("     early stopping", flush=True); break
    if bsd: model.load_state_dict(bsd)
    torch.save(dict(model_state_dict=bsd, val_f1=best), CKPT)

    # Evaluación
    ids, vP, vT = infer(model, dl_va); y = np.argmax(vT, 1)
    cands = sorted(((wf1(y, dec(vP,tj,td)),float(tj),float(td)) for tj in np.arange(0.12,0.50,0.02) for td in np.arange(0.22,0.60,0.02)), reverse=True)
    bestT=None
    for w,tj,td in cands[:8]:
        ic = icm_hard(ids, y, dec(vP,tj,td)); sc = 0.5*w + 0.5*(ic+1)/2
        if bestT is None or sc>bestT[0]: bestT=(sc,tj,td)
    _, tjE, tdE = bestT
    icm_argmax = icm_hard(ids, y, np.argmax(vP,1))
    icm_thr    = icm_hard(ids, y, dec(vP, tjE, tdE))
    icms_raw   = icmsoft(ids, vT, vP)
    oh = np.eye(3)[y]; platt = []
    for c in range(3):
        lg = np.log(np.clip(vP[:,c],EPS,1-EPS)/np.clip(1-vP[:,c],EPS,1-EPS))
        lr = LogisticRegression().fit(lg.reshape(-1,1), oh[:,c]); platt.append((float(lr.coef_[0,0]), float(lr.intercept_[0])))
    vP_cal = np.zeros_like(vP)
    for c,(a,b) in enumerate(platt):
        lg = np.log(np.clip(vP[:,c],EPS,1-EPS)/np.clip(1-vP[:,c],EPS,1-EPS)); vP_cal[:,c] = 1/(1+np.exp(-(a*lg+b)))
    vP_cal = vP_cal/vP_cal.sum(1,keepdims=True).clip(min=EPS)
    icms_platt = icmsoft(ids, vT, vP_cal)
    fc = f1_score(y, dec(vP, tjE, tdE), average=None, labels=[0,1,2])
    print(f"\n=== RESULTADOS Task 2.2 EXP-B (Longformer max_length={MAX_TOK}) ===", flush=True)
    print(f"  thr ponderado: t_JUDG={tjE:.3f} t_DIR={tdE:.3f}", flush=True)
    print(f"  HARD: ICM(argmax)={icm_argmax:+.4f}  ICM(thr)={icm_thr:+.4f}  F1macro(thr)={f1_score(y,dec(vP,tjE,tdE),average='macro',labels=[0,1,2]):.4f}  F1[N/D/J]={fc[0]:.3f}/{fc[1]:.3f}/{fc[2]:.3f}", flush=True)
    print(f"  SOFT: ICMSoft(raw)={icms_raw:+.4f}  ICMSoft(Platt)={icms_platt:+.4f}", flush=True)


if __name__ == "__main__":
    main()
