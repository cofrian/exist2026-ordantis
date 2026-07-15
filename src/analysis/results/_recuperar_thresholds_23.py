"""PASO 1 — Recuperar los thresholds OFICIALES del run 2.3 sobre el principal.
Reutiliza EXACTAMENTE los helpers de _full_eval_task23.py (gold_hard_from_soft,
pred_from_probs, pyevall_hard_full/soft_full, find_best_thr) sobre las probs crudas
ya guardadas en preds_val_vista_e_task23_best.csv. No re-infiere.
Gate de sexismo del principal = max prob de categoria (identico a eval_main23.py)."""
import os, sys, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../Trabajo_LNR
sys.path.insert(0, ROOT)
import _full_eval_task23 as F                      # helpers oficiales + monkeypatch sigma
sys.path.insert(0, HERE)
import _common as K

CATS = F.CATS
g = K.gold()

def load_probs(name):
    p = os.path.join(HERE, f"preds_val_{name}.csv")
    ids, PC = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            ids.append(row["id_meme"])
            PC.append([float(row["P_IDEOLOGICAL"]), float(row["P_STEREOTYPING"]),
                       float(row["P_OBJECTIFICATION"]), float(row["P_SEXUAL_VIOLENCE"]),
                       float(row["P_MISOGYNY_NSV"])])
    return ids, np.array(PC, np.float32)

ids, pc = load_probs("vista_e_task23_best")
T  = np.array([g["t23_soft"][i] for i in ids], np.float32)     # gold soft por categoria [5]
SXg = np.array([g["t23_sex"][i] for i in ids], np.float32)     # gold sexismo (proporcion)
ps = pc.max(1)                                                 # gate de sexismo del modelo (principal)

gh = F.gold_hard_from_soft(T, SXg)

# --- Barrido OFICIAL (find_best_thr): tsex in [0.30,0.66) step .04 ; tcat uniforme [0.05,0.55) step .05
print("Barrido oficial (tsex x tcat uniforme), objetivo 0.5*Fmacro + 0.5*ICMNorm ...", flush=True)
resultados = []
for tsex in np.arange(0.30, 0.66, 0.04):
    for tc in np.arange(0.05, 0.55, 0.05):
        pr = F.pred_from_probs(ps, pc, float(tsex), np.full(5, float(tc)))
        icm, icmn, fm = F.pyevall_hard_full(ids, gh, pr)
        resultados.append((float(tsex), float(tc), icm, icmn, fm, 0.5*fm + 0.5*icmn))

# Top-5 por ICM oficial
por_icm = sorted(resultados, key=lambda r: r[2], reverse=True)[:5]
# Top-1 por el criterio oficial (0.5*Fmacro+0.5*ICMNorm) = lo que realmente eligio el paper
por_crit = sorted(resultados, key=lambda r: r[5], reverse=True)[0]

print("\n=== TOP-5 combinaciones por ICM oficial (vista_e_task23_best) ===")
print(f"{'tsex':>5} {'tcat':>5} {'ICM':>9} {'ICMNorm':>9} {'Fmacro':>8} {'0.5F+0.5N':>10}")
for tsex, tc, icm, icmn, fm, sc in por_icm:
    print(f"{tsex:5.2f} {tc:5.2f} {icm:+9.4f} {icmn:9.4f} {fm:8.4f} {sc:10.4f}")

print("\n=== Combinacion elegida por el criterio OFICIAL (0.5*Fmacro+0.5*ICMNorm) ===")
tsex, tc, icm, icmn, fm, sc = por_crit
print(f"tsex={tsex:.2f}  tcat={tc:.2f}  ->  ICM={icm:+.4f}  ICMNorm={icmn:.4f}  Fmacro={fm:.4f}")

# ICMSoft del principal (no depende de threshold)
icmsoft, icmsoftn = F.pyevall_soft_full(ids, T, SXg, ps, pc)
print(f"ICMSoft={icmsoft:+.4f}  ICMSoftNorm={icmsoftn:.4f}")

# guardar para el .md
with open(os.path.join(HERE, "_thr_top5_principal.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tsex","tcat","ICM","ICMNorm","Fmacro","crit_0.5F+0.5N"])
    for r in por_icm: w.writerow([f"{r[0]:.2f}",f"{r[1]:.2f}",f"{r[2]:.4f}",f"{r[3]:.4f}",f"{r[4]:.4f}",f"{r[5]:.4f}"])
    w.writerow([]); w.writerow(["ELEGIDA_CRITERIO_OFICIAL"])
    w.writerow([f"{por_crit[0]:.2f}",f"{por_crit[1]:.2f}",f"{por_crit[2]:.4f}",f"{por_crit[3]:.4f}",f"{por_crit[4]:.4f}",f"{por_crit[5]:.4f}"])
print("\n-> _thr_top5_principal.csv")
