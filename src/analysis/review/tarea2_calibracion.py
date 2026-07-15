"""TAREA 2 - Calibracion sobre VALIDACION (598) con los checkpoints entregados.
Calcula ECE (10 bins), MCE, Brier; reliability diagrams (hard y soft); ECE por clase
one-vs-rest (3 de 2.2, 5 de 2.3); y compara ANTES vs DESPUES de calibrar
(Platt por clase en 2.2/2.3; blend 0.6/0.4 con Gemini y temperature scaling en 2.1).

Gold hard = voto mayoritario (umbral 0.5). Reliability 'soft' = prob predicha vs
proporcion real de anotadores. NO se toca test (sin gold publico).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras"); os.makedirs(FIG, exist_ok=True)
import config as C, data as D

CSV_ROWS = []
def addrow(**k): CSV_ROWS.append(k)

# ---------------- metricas de calibracion ----------------
def bin_stats(p, y, nbins=10):
    """Devuelve por bin: (n, conf=mean p, acc=mean y). y puede ser 0/1 (hard) o [0,1] (soft)."""
    edges = np.linspace(0, 1, nbins + 1)
    out = []
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < nbins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            out.append((0, np.nan, np.nan)); continue
        out.append((int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out, edges

def ece_mce(p, y, nbins=10):
    stats, _ = bin_stats(p, y, nbins)
    n = len(p); ece = 0.0; mce = 0.0
    for cnt, conf, acc in stats:
        if cnt == 0: continue
        gap = abs(acc - conf)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
    return ece, mce

def brier(p, y):
    return float(np.mean((p - y) ** 2))

def reliability_plot(ax, p, y, title, nbins=10):
    stats, edges = bin_stats(p, y, nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    confs = [s[1] for s in stats]; accs = [s[2] for s in stats]; cnts = [s[0] for s in stats]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    xs = [c for c, s in zip(centers, stats) if s[0] > 0]
    ys = [s[2] for s in stats if s[0] > 0]
    cf = [s[1] for s in stats if s[0] > 0]
    ax.bar(centers, [s[2] if s[0] > 0 else 0 for s in stats], width=1.0/nbins*0.9,
           alpha=0.6, edgecolor="black", label="accuracy")
    ax.plot(cf, ys, "o-", color="C3", label="acc vs conf")
    e, mc = ece_mce(p, y, nbins)
    ax.set_title(f"{title}\nECE={e:.3f} MCE={mc:.3f}", fontsize=9)
    ax.set_xlabel("confianza predicha"); ax.set_ylabel("frec. real")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=6)

def block(name, subtask, p, y_hard, y_soft, cls="GLOBAL", method="raw"):
    e_h, m_h = ece_mce(p, y_hard); b_h = brier(p, y_hard)
    e_s, m_s = ece_mce(p, y_soft); b_s = brier(p, y_soft)
    addrow(subtask=subtask, model=name, clase=cls, metodo=method,
           ECE_hard=round(e_h, 4), MCE_hard=round(m_h, 4), Brier_hard=round(b_h, 4),
           ECE_soft=round(e_s, 4), Brier_soft=round(b_s, 4), n=len(p))
    return e_h, e_s

# ---------------- 2.1 binario ----------------
def do_task21():
    from dataset import MemeDataset, make_collate, to_device
    from models import MemeClassifier
    print("\n### 2.1 ###", flush=True)
    g = json.load(open(os.path.join(C.PRE_DIR, "gemini_predictions.json")))
    caps, gprob = {}, {}
    for mid, v in g.items():
        if isinstance(v, dict):
            d = (v.get("description") or "").strip(); a = (v.get("sexism_analysis") or "").strip()
            parts = []
            if d: parts.append("Description: " + d)
            if a: parts.append("Sexism Analysis: " + a)
            if parts: caps[str(mid)] = " ".join(parts)
            try: gprob[str(mid)] = float(v["task2_1"]["sexist_probability"])
            except Exception: pass
    sp = D.load_split()
    mcfg = dict(text=True, image=False, et=False, hr=False, eeg=True, caption=True, set_pool=True, emotions=True)
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL); coll = make_collate(tok)
    model = MemeClassifier(mcfg)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, "M3_vista_E_best.pt"),
                                     map_location="cpu", weights_only=False)["model_state_dict"])
    model.to(C.DEVICE).eval()
    ex = sp["val"]
    ds = MemeDataset(ex, tok, vit_emb={e["id"]: np.zeros(768, np.float32) for e in ex}, captions=caps, use_caption=True)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=coll)
    ids, lo, tg = [], [], []
    with torch.no_grad():
        for b in dl:
            b = to_device(b, C.DEVICE)
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                l = model(b)
            ids += b["id"]; lo.append(l.float().cpu().numpy()); tg.append(b["soft"].cpu().numpy())
    lo = np.concatenate(lo); tg = np.concatenate(tg)
    pE = 1/(1+np.exp(-lo)); yh = (tg >= 0.5).astype(int)
    pG = np.array([gprob.get(str(i), 0.5) for i in ids])
    pEG = 0.6*pE + 0.4*pG
    # temperature scaling (fit en val, in-sample -> se anota)
    import evaluation_utils as EU
    T = EU.fit_temperature(lo, tg); pT = EU.apply_temperature(lo, T)
    block("Vista E (raw)", "2.1", pE, yh, tg, method="raw")
    block("Vista E (temperature T=%.3f)" % T, "2.1", pT, yh, tg, method="temperature")
    block("Gemini crudo", "2.1", pG, yh, tg, method="raw")
    block("Ensemble 0.6E+0.4G", "2.1", pEG, yh, tg, method="blend_0.6/0.4")
    # figuras: reliability hard y soft de Vista E raw vs temp vs ensemble
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    reliability_plot(axs[0,0], pE, yh, "2.1 Vista E raw (hard)")
    reliability_plot(axs[0,1], pT, yh, "2.1 Vista E temp (hard)")
    reliability_plot(axs[0,2], pEG, yh, "2.1 Ensemble 0.6/0.4 (hard)")
    reliability_plot(axs[1,0], pE, tg, "2.1 Vista E raw (soft)")
    reliability_plot(axs[1,1], pT, tg, "2.1 Vista E temp (soft)")
    reliability_plot(axs[1,2], pG, yh, "2.1 Gemini crudo (hard)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability_2_1.png"), dpi=110); plt.close(fig)
    print("  fig: figuras/reliability_2_1.png", flush=True)

# ---------------- Platt por clase ----------------
def platt_fit_apply(P, Y):
    cal = np.zeros_like(P)
    for i in range(P.shape[1]):
        p = np.clip(P[:, i], 1e-7, 1-1e-7); z = np.log(p/(1-p)).reshape(-1,1)
        if len(np.unique(Y[:, i])) < 2:
            cal[:, i] = P[:, i]; continue
        lr = LogisticRegression(C=1.0); lr.fit(z, Y[:, i])
        cal[:, i] = lr.predict_proba(z)[:, 1]
    return cal

# ---------------- 2.2 3-clases ----------------
def do_task22():
    import task22 as T22
    print("\n### 2.2 ###", flush=True)
    splits = T22.load_task22()
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    dl = DataLoader(T22.DS22(splits["val"], tok), batch_size=64, shuffle=False, collate_fn=T22.collate(tok), num_workers=4)
    model = T22.VistaE22().to(C.DEVICE)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, "vista_e_task22_best.pt"),
                                     map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
    ids, P, Tt = T22.infer(model, dl)
    Yh = np.eye(3)[np.argmax(Tt, 1)]   # gold hard one-hot (argmax = voto mayoritario)
    Pcal = platt_fit_apply(P, Yh)
    names = ["NO", "DIRECT", "JUDGEMENTAL"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    e_raw, e_cal = [], []
    for i, nm in enumerate(names):
        er,_ = block("VistaE22 raw", "2.2", P[:, i], Yh[:, i], Tt[:, i], cls=nm, method="raw")
        ec,_ = block("VistaE22 Platt", "2.2", Pcal[:, i], Yh[:, i], Tt[:, i], cls=nm, method="platt_per_class")
        e_raw.append(er); e_cal.append(ec)
        reliability_plot(axs[0, i], P[:, i], Yh[:, i], f"2.2 {nm} raw (hard)")
        reliability_plot(axs[1, i], Pcal[:, i], Yh[:, i], f"2.2 {nm} Platt (hard)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability_2_2_perclass.png"), dpi=110); plt.close(fig)
    # ECE macro (media OvR)
    addrow(subtask="2.2", model="VistaE22 raw", clase="MACRO_OvR", metodo="raw",
           ECE_hard=round(float(np.mean(e_raw)), 4), MCE_hard="", Brier_hard="", ECE_soft="", Brier_soft="", n=len(ids))
    addrow(subtask="2.2", model="VistaE22 Platt", clase="MACRO_OvR", metodo="platt_per_class",
           ECE_hard=round(float(np.mean(e_cal)), 4), MCE_hard="", Brier_hard="", ECE_soft="", Brier_soft="", n=len(ids))
    print(f"  ECE macro OvR: raw={np.mean(e_raw):.4f} -> Platt={np.mean(e_cal):.4f}", flush=True)
    print("  fig: figuras/reliability_2_2_perclass.png", flush=True)

# ---------------- 2.3 5-categorias (modelo principal, desde cache) ----------------
def do_task23():
    print("\n### 2.3 ###", flush=True)
    import task23 as T23
    cache = os.path.join(HERE, "cache_main23_val.npz")
    dat = np.load(cache, allow_pickle=True)
    P = dat["P"]; Tt = dat["T"]  # [N,5] probs, [N,5] soft cat gold
    Yh = (Tt > 1/6 + 1e-9).astype(int)  # gold hard cat: mayoria de anotadores (>1/6 ~ al menos 1 de 6)
    Pcal = platt_fit_apply(P, Yh)
    cats = T23.CATS
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    e_raw, e_cal = [], []
    for i, nm in enumerate(cats):
        er,_ = block("VistaE23 raw", "2.3", P[:, i], Yh[:, i], Tt[:, i], cls=nm, method="raw")
        ec,_ = block("VistaE23 Platt", "2.3", Pcal[:, i], Yh[:, i], Tt[:, i], cls=nm, method="platt_per_class")
        e_raw.append(er); e_cal.append(ec)
        reliability_plot(axs[0, i], P[:, i], Yh[:, i], f"2.3 {nm[:12]} raw")
        reliability_plot(axs[1, i], Pcal[:, i], Yh[:, i], f"2.3 {nm[:12]} Platt")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability_2_3_percat.png"), dpi=110); plt.close(fig)
    addrow(subtask="2.3", model="VistaE23 raw", clase="MACRO_OvR", metodo="raw",
           ECE_hard=round(float(np.mean(e_raw)), 4), MCE_hard="", Brier_hard="", ECE_soft="", Brier_soft="", n=P.shape[0])
    addrow(subtask="2.3", model="VistaE23 Platt", clase="MACRO_OvR", metodo="platt_per_class",
           ECE_hard=round(float(np.mean(e_cal)), 4), MCE_hard="", Brier_hard="", ECE_soft="", Brier_soft="", n=P.shape[0])
    print(f"  ECE macro OvR: raw={np.mean(e_raw):.4f} -> Platt={np.mean(e_cal):.4f}", flush=True)
    print("  fig: figuras/reliability_2_3_percat.png", flush=True)

if __name__ == "__main__":
    import csv
    do_task21()
    do_task22()
    do_task23()
    cols = ["subtask","model","clase","metodo","ECE_hard","MCE_hard","Brier_hard","ECE_soft","Brier_soft","n"]
    out = os.path.join(HERE, "calibracion.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(CSV_ROWS)
    print("\nCSV:", out, f"({len(CSV_ROWS)} filas)")
