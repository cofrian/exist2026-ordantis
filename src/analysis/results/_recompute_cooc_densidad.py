"""Punto 2 — Recalcula co-ocurrencia, densidad multi-etiqueta y freq_gold(Gemini) con el GOLD OFICIAL
(Modulo 1), el MISMO que usan los F1 por categoria del Bloque 2 (cache_23_gate.npz -> gold_hard_from_soft,
categoria presente si >1 anotador == soft>1/6, con los 598 memes como categorizables SX=1).
Sobrescribe cooc_gold_2_3.csv, densidad_multietiqueta_gold.csv y la columna freq_gold de
gemini_prob_media_por_categoria.csv."""
import os, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ["IDEOLOGICAL-INEQUALITY","STEREOTYPING-DOMINANCE","OBJECTIFICATION","SEXUAL-VIOLENCE","MISOGYNY-NON-SEXUAL-VIOLENCE"]

# ---- gold oficial identico al de Bloques 2/5 ----
Z = np.load(os.path.join(HERE, "cache_23_gate.npz"), allow_pickle=True)
gold_ids = [str(x) for x in Z["gold__ids"]]; T = Z["gold__T"]; SX = Z["gold__SX"]
def gold_hard_from_soft(soft5, sex):
    G = np.zeros((len(sex), 5), int)
    for i in range(len(sex)):
        if sex[i] < 0.5: continue                      # "NO" -> sin categorias (aqui SX=1 siempre)
        cats = [c for c in range(5) if soft5[i, c] > (1.0/6 + 1e-9)]
        if not cats: cats = [int(np.argmax(soft5[i]))] # fallback formato (>=1), igual que el oficial
        for c in cats: G[i, c] = 1
    return G
G = gold_hard_from_soft(T, SX)

# ---- co-ocurrencia (conteos) ----
CO = G.T @ G   # diagonal = freq por cat; fuera de diagonal = co-ocurrencias
with open(os.path.join(HERE, "cooc_gold_2_3.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow([""]+CATS)
    for i in range(5): w.writerow([CATS[i]]+[int(CO[i, j]) for j in range(5)])
print("-> cooc_gold_2_3.csv (gold oficial)")
print("   freq por categoria (diagonal):", {CATS[i]: int(CO[i,i]) for i in range(5)})

# ---- densidad multi-etiqueta ----
ncat = G.sum(1)
import collections
dist = collections.Counter(int(x) for x in ncat)
with open(os.path.join(HERE, "densidad_multietiqueta_gold.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["num_categorias", "num_memes"])
    for k in sorted(dist): w.writerow([k, dist[k]])
print("-> densidad_multietiqueta_gold.csv:", dict(sorted(dist.items())),
      f"| media={ncat.mean():.3f} cat/meme")

# ---- freq_gold de gemini_prob_media (conservar prob_media, actualizar freq_gold) ----
gp = os.path.join(HERE, "gemini_prob_media_por_categoria.csv")
if os.path.exists(gp):
    rows = list(csv.DictReader(open(gp)))
    freq = {CATS[i]: int(CO[i, i]) for i in range(5)}
    for r in rows:
        if r["categoria"] in freq: r["freq_gold"] = freq[r["categoria"]]
    with open(gp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("-> gemini_prob_media_por_categoria.csv (freq_gold -> gold oficial; prob_media intacta)")
