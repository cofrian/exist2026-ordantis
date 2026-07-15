"""Consolida metricas_18_modelos.csv a partir de los LOGS REALES ya generados
(no re-ejecuta modelos, no inventa numeros). Parsea con regex el stdout capturado."""
import os, re, csv

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
def L(name): return open(os.path.join(LOGS, name), encoding="utf-8", errors="ignore").read()

rows = []
def add(**k): rows.append(k)

# ---------- 2.1 principales (Vista E, Gemini, Ensemble) ----------
for m in re.finditer(r"^\s*(Vista E|Gemini 3-flash crudo|Ensemble 0\.6\*E \+ 0\.4\*Gemini)\s+"
                     r"F1\+=([\-0-9.]+)\s+AUC=([\-0-9.]+)\s+ICM=([+\-0-9.]+)\s+ICMSoft=([+\-0-9.]+)\s+thr=([0-9.]+)",
                     L("t1_task21_rerun.log"), re.M):
    name, f1, auc, icm, icms, thr = m.groups()
    add(subtask="2.1", model=name.strip(), checkpoint={"Vista E":"M3_vista_E_best.pt","Gemini 3-flash crudo":"(Gemini API cache)","Ensemble 0.6*E + 0.4*Gemini":"M3_vista_E_best.pt + Gemini"}[name.strip()],
        F1=f1, F1_name="F1_pos", AUC=auc, ICM=icm, ICMNorm="", ICMSoft=icms, ICMSoftNorm="", extra=f"thr={thr}", source="t1_task21_rerun.log")

# ---------- 2.1 variantes _alt (desde CSV) ----------
p21 = os.path.join(HERE, "task21_variants.csv")
if os.path.exists(p21):
    for r in csv.DictReader(open(p21)):
        add(subtask="2.1", model=r["model"], checkpoint=r["checkpoint"], F1=f'{float(r["F1_pos"]):.4f}', F1_name="F1_pos",
            AUC=f'{float(r["AUC"]):.4f}', ICM=f'{float(r["ICM"]):+.4f}', ICMNorm="", ICMSoft=f'{float(r["ICMSoft"]):+.4f}',
            ICMSoftNorm="", extra=f'thr={r["thr"]}', source="task21_variants.csv")

# ---------- 2.2 principal (varias decodificaciones, _reeval) ----------
t22 = L("t1_task22_reeval.log")
for m in re.finditer(r"^\s*HARD (.+?)\s{2,}ICM=([+\-0-9.]+)\s+ICMNorm=([0-9.]+)\s+F1macro=([0-9.]+)\s+F1\[N/D/J\]=([0-9./]+)", t22, re.M):
    name, icm, icmn, f1m, f1c = m.groups()
    add(subtask="2.2", model=f"[main] HARD {name.strip()}", checkpoint="vista_e_task22_best.pt", F1=f1m, F1_name="F1_macro",
        AUC="", ICM=icm, ICMNorm=icmn, ICMSoft="", ICMSoftNorm="", extra=f"F1[N/D/J]={f1c}", source="t1_task22_reeval.log")
for m in re.finditer(r"^\s*SOFT (.+?)\s{2,}ICMSoft=([+\-0-9.]+)\s+ICMSoftNorm=([0-9.]+)\s+CE=([0-9.]+)\s+F1macro\(argmax\)=([0-9.]+)", t22, re.M):
    name, icms, icmsn, ce, f1m = m.groups()
    add(subtask="2.2", model=f"[main] SOFT {name.strip()}", checkpoint="vista_e_task22_best.pt", F1=f1m, F1_name="F1_macro_argmax",
        AUC="", ICM="", ICMNorm="", ICMSoft=icms, ICMSoftNorm=icmsn, extra=f"CE={ce}", source="t1_task22_reeval.log")

# ---------- 2.2 variantes _alt (desde CSV) ----------
p22 = os.path.join(HERE, "task22_variants.csv")
if os.path.exists(p22):
    for r in csv.DictReader(open(p22)):
        add(subtask="2.2", model=r["model"], checkpoint=r["checkpoint"], F1=f'{float(r["F1macro_thr"]):.4f}', F1_name="F1_macro_thr",
            AUC="", ICM=f'{float(r["ICM"]):+.4f}', ICMNorm=f'{float(r["ICMNorm"]):.4f}', ICMSoft=f'{float(r["ICMSoft"]):+.4f}',
            ICMSoftNorm=f'{float(r["ICMSoftNorm"]):.4f}',
            extra=f'argmax={float(r["F1macro_argmax"]):.4f}; F1[N/D/J]={float(r["F1_NO"]):.2f}/{float(r["F1_DIRECT"]):.2f}/{float(r["F1_JUDG"]):.2f}',
            source="task22_variants.csv")

# ---------- 2.3 (7 checkpoints + Gemini) ----------
def parse_23_block(text, source):
    out = []
    # bloques delimitados por '=== N. <label> ==='
    blocks = re.split(r"^=== (\d+)\. (.+?) ===\s*$", text, flags=re.M)
    # blocks: [pre, num, label, body, num, label, body, ...]
    for i in range(1, len(blocks), 3):
        num, label, body = blocks[i], blocks[i+1], blocks[i+2]
        f1mi = re.search(r"F1 micro \(cat @ 0\.5\):\s*([0-9.]+)", body)
        f1ma = re.search(r"F1macro(?: \(con thr\))?:\s*([0-9.]+)", body)
        icm  = re.search(r"ICM hard(?: \(con thr\))?:\s*([+\-0-9.]+)\s+ICMNorm:\s*([0-9.]+)", body)
        icms = re.search(r"ICMSoft:\s*([+\-0-9.]+)\s+ICMSoftNorm:\s*([0-9.]+)", body)
        thr  = re.search(r"thr_sex=([0-9.]+)\s+thr_cat=([0-9.]+)", body)
        if not (f1ma and icm and icms):  # bloque con ERROR -> saltar
            continue
        out.append((num, label.strip(), f1mi.group(1) if f1mi else "",
                    f1ma.group(1), icm.group(1), icm.group(2), icms.group(1), icms.group(2),
                    (f"thr_sex={thr.group(1)},thr_cat={thr.group(2)}" if thr else ""), source))
    return out

CKPT23 = {
 "1. Vista E-2.3 ORIGINAL": "vista_e_task23_best.pt",
 "2. Vista E-2.3 max=512": "vista_e_task23_max512_best.pt",
 "3. Vista E-2.3 Longformer": "vista_e_task23_longformer_best.pt",
 "4. Vista E-2.3 max=512_v2": "vista_e_task23_max512_v2_best.pt",
 "5. Vista E-2.3 Longformer_v2": "vista_e_task23_longformer_v2_best.pt",
 "6. Vista E-2.3 max=512_R": "vista_e_task23_max512_R_best.pt",
 "7. Vista E-2.3 Longformer_R": "vista_e_task23_longformer_R_best.pt",
 "8. Gemini 3-flash crudo": "(Gemini API cache)",
}
def ckpt_for(num, label):
    key = f"{num}. " + label.split("(")[0].strip().rstrip()
    for k, v in CKPT23.items():
        if key.startswith(k): return v
    return ""

seen23 = set()
# main (item 1) desde su log dedicado
for tup in parse_23_block(L("t1_task23_main.log"), "t1_task23_main.log"):
    num, label, f1mi, f1ma, icm, icmn, icms, icmsn, extra, src = tup
    ck = ckpt_for(num, label); seen23.add(num)
    add(subtask="2.3", model=f"Vista E-2.3 ORIGINAL (zip)", checkpoint=ck, F1=f1ma, F1_name="F1_macro",
        AUC="", ICM=icm, ICMNorm=icmn, ICMSoft=icms, ICMSoftNorm=icmsn, extra=f"F1micro={f1mi}; {extra}", src2="", source=src)
# items 2..8 desde el full
for tup in parse_23_block(L("t1_task23_full_fixed.log"), "t1_task23_full_fixed.log"):
    num, label, f1mi, f1ma, icm, icmn, icms, icmsn, extra, src = tup
    if num in seen23: continue
    seen23.add(num)
    ck = ckpt_for(num, label)
    add(subtask="2.3", model=(label if not label.startswith("Gemini") else "Gemini 3-flash crudo (2.3)"),
        checkpoint=ck, F1=f1ma, F1_name="F1_macro", AUC="", ICM=icm, ICMNorm=icmn, ICMSoft=icms, ICMSoftNorm=icmsn,
        extra=f"F1micro={f1mi}; {extra}", source=src)

# ---------- escribir CSV ----------
cols = ["subtask","model","checkpoint","F1","F1_name","AUC","ICM","ICMNorm","ICMSoft","ICMSoftNorm","extra","source"]
out = os.path.join(HERE, "metricas_18_modelos.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)
print(f"Filas: {len(rows)}  ->  {out}")
for r in rows:
    print(f"  [{r['subtask']}] {r['model']:<44} F1={r.get('F1',''):<7} ICM={r.get('ICM',''):<9} ICMSoft={r.get('ICMSoft','')}")
