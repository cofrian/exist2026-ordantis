"""TAREA 3 - Analisis de errores del modelo 2.3 PRINCIPAL (vista_e_task23_best.pt) vs gold,
sobre VALIDACION. Usa el cache cache_main23_val.npz (ids, P[N,5], T[N,5], SX, tsex, tcat).

Produce:
  - errores_2_3.csv         : por categoria -> freq gold, F1, precision, recall, TP/FP/FN/TN
  - cooc_gold_2_3.csv       : co-ocurrencia 5x5 en gold
  - cooc_pred_2_3.csv       : co-ocurrencia 5x5 en predicciones
  - confus_goldcat_predcat_2_3.csv : fila=cat gold presente -> reparto de cats predichas
  - figuras/confusion_2_3.png
  - errores_2_3_resumen.md  : lectura (rareza vs confusion de frontera)
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras"); os.makedirs(FIG, exist_ok=True)
import config as C
import task23 as T23
CATS = T23.CATS

dat = np.load(os.path.join(HERE, "cache_main23_val.npz"), allow_pickle=True)
ids = dat["ids"]; P = dat["P"]; T = dat["T"]; SX = dat["SX"]
tsex = float(dat["tsex"]); tcat = dat["tcat"].astype(float)
ps = P.max(axis=1)  # compuerta de sexista del modelo principal (misma que en eval_main23)

N = len(ids)
# gold binario por categoria: categoria presente si el meme es sexista (SX>=0.5) y
# la proporcion de anotadores que marca la cat supera 1/6 (al menos ~1 de 6). Coherente
# con gold_hard_from_soft de _full_eval_task23.py.
G = np.zeros((N, 5), int)
for i in range(N):
    if SX[i] >= 0.5:
        for c in range(5):
            if T[i, c] > 1/6 + 1e-9: G[i, c] = 1
# prediccion binaria: compuerta sexista (ps>=tsex) y cat sobre su umbral
Pr = np.zeros((N, 5), int)
for i in range(N):
    if ps[i] >= tsex:
        for c in range(5):
            if P[i, c] >= tcat[c]: Pr[i, c] = 1

# ---- metricas por categoria ----
import csv
rows = []
for c in range(5):
    g, p = G[:, c], Pr[:, c]
    tp = int(((g==1)&(p==1)).sum()); fp = int(((g==0)&(p==1)).sum())
    fn = int(((g==1)&(p==0)).sum()); tn = int(((g==0)&(p==0)).sum())
    f1 = f1_score(g, p, zero_division=0); pr = precision_score(g, p, zero_division=0); rc = recall_score(g, p, zero_division=0)
    rows.append(dict(categoria=CATS[c], freq_gold=int(g.sum()), freq_pred=int(p.sum()),
                     F1=round(f1,4), precision=round(pr,4), recall=round(rc,4),
                     TP=tp, FP=fp, FN=fn, TN=tn))
with open(os.path.join(HERE, "errores_2_3.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- co-ocurrencia gold y pred (5x5, conteo de memes con ambas cats) ----
def cooc(M):
    C5 = np.zeros((5,5), int)
    for i in range(len(M)):
        act = np.where(M[i]==1)[0]
        for a in act:
            for b in act:
                C5[a,b]+=1
    return C5
CG, CP = cooc(G), cooc(Pr)
def save_mat(mat, path, labels=CATS):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow([""]+labels)
        for i,l in enumerate(labels): w.writerow([l]+list(mat[i]))
save_mat(CG, os.path.join(HERE, "cooc_gold_2_3.csv"))
save_mat(CP, os.path.join(HERE, "cooc_pred_2_3.csv"))

# ---- confusion gold-cat -> pred-cat: para memes con cat gold=g, cuantos predicen cat p ----
# fila g normalizada por nº de memes con gold cat g.
CF = np.zeros((5,5), float)
for g in range(5):
    idx = np.where(G[:, g]==1)[0]
    if len(idx)==0: continue
    for p in range(5):
        CF[g,p] = Pr[idx, p].sum()/len(idx)
save_mat((CF*1000).astype(int)/1000.0, os.path.join(HERE, "confus_goldcat_predcat_2_3.csv"))

# ---- figura: 3 heatmaps ----
fig, axs = plt.subplots(1,3, figsize=(20,6))
for ax, M, ttl, fmt in [(axs[0], CG, "Co-ocurrencia GOLD", "d"),
                         (axs[1], CP, "Co-ocurrencia PRED", "d"),
                         (axs[2], CF, "P(pred cat | gold cat)", ".2f")]:
    im = ax.imshow(M, cmap="Blues"); ax.set_title(ttl, fontsize=11)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([c[:10] for c in CATS], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([c[:10] for c in CATS], fontsize=8)
    for i in range(5):
        for j in range(5):
            v = M[i,j]; ax.text(j,i, (f"{int(v)}" if fmt=="d" else f"{v:.2f}"), ha="center", va="center", fontsize=7,
                                color="white" if v> (M.max()*0.6) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_2_3.png"), dpi=110); plt.close(fig)

# ---- resumen texto ----
order = sorted(rows, key=lambda r: r["freq_gold"])
lines = ["# TAREA 3 - Analisis de errores 2.3 (modelo principal, validacion n=%d)\n" % N,
         "Umbrales usados (optimizados sobre val en eval_main23): thr_sex=%.2f, thr_cat=%.2f\n" % (tsex, tcat[0]),
         "## F1 por categoria (ordenado por frecuencia gold ascendente)\n",
         "| categoria | freq_gold | F1 | precision | recall | FP | FN |",
         "|---|---|---|---|---|---|---|"]
for r in order:
    lines.append(f"| {r['categoria']} | {r['freq_gold']} | {r['F1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['FP']} | {r['FN']} |")
# correlacion frecuencia vs F1
fr = np.array([r["freq_gold"] for r in rows], float); f1s = np.array([r["F1"] for r in rows], float)
corr = float(np.corrcoef(fr, f1s)[0,1]) if len(set(fr))>1 else float("nan")
lines += ["",
          f"Correlacion frecuencia_gold vs F1 (Pearson, n=5 cats): r = {corr:+.3f}.",
          "Si r es alto y positivo -> el F1 bajo se explica por RAREZA (pocas muestras).",
          "Las off-diagonales de 'P(pred cat | gold cat)' revelan CONFUSION DE FRONTERA",
          "(el modelo predice otra categoria cuando la gold es X).",
          "", "## Confusiones de frontera mas fuertes (off-diagonal de P(pred|gold))"]
offs = []
for g in range(5):
    for p in range(5):
        if g!=p: offs.append((CF[g,p], CATS[g], CATS[p]))
for v,g,p in sorted(offs, reverse=True)[:6]:
    lines.append(f"- gold={g} -> predice tambien {p}: {v:.2f}")
open(os.path.join(HERE, "errores_2_3_resumen.md"), "w").write("\n".join(lines))

print("Escrito: errores_2_3.csv, cooc_gold/pred, confus_goldcat_predcat, figuras/confusion_2_3.png, errores_2_3_resumen.md")
print("\nF1 por categoria (freq asc):")
for r in order:
    print(f"  {r['categoria']:<30} freq={r['freq_gold']:<4} F1={r['F1']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f}  FP={r['FP']} FN={r['FN']}")
print(f"\nCorrelacion freq vs F1: r={corr:+.3f}")
