"""Helpers oficiales de decode/eval de Task 2.3 (copiados verbatim de _full_eval_task23.py).
Modulo SIN efectos colaterales (solo el monkeypatch sigma=0). Importable con seguridad."""
import os, json, tempfile, math
import numpy as np
from statistics import NormalDist
from pyevall.metrics.metrics import ICMSoft
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

def _safe(self, t, c):
    if t is None or not t[0]: return 0
    if t[0] not in self.gold_average: return -math.log2(1/len(c.gold_df))
    if t[1] == 0.0: return 0.0
    sigma = max(float(self.gold_deviation[t[0]]), 1e-9)
    try: prob = 1 - NormalDist(mu=self.gold_average[t[0]], sigma=sigma).cdf(t[1])
    except: return -math.log2(1/len(c.gold_df))
    if prob <= 0.0: return -math.log2(1/len(c.gold_df))
    return -math.log2(prob)
ICMSoft.get_prob_class = _safe

CATS = ["IDEOLOGICAL-INEQUALITY","STEREOTYPING-DOMINANCE","OBJECTIFICATION","SEXUAL-VIOLENCE","MISOGYNY-NON-SEXUAL-VIOLENCE"]
HIER = {"YES": CATS, "NO": []}; TC = "EXIST2025"

def gold_hard_from_soft(soft5, sex):
    y = []
    for i in range(len(sex)):
        if sex[i] < 0.5: y.append(["NO"]); continue
        cats = [CATS[c] for c in range(5) if soft5[i, c] > (1.0/6 + 1e-9)]
        y.append(cats if cats else [CATS[int(np.argmax(soft5[i]))]])
    return y

def pred_from_probs(ps, pc, thr_sex=0.5, thr_cat=None):
    if thr_cat is None: thr_cat = np.full(5, 0.5)
    out = []
    for i in range(len(ps)):
        if ps[i] < thr_sex: out.append(["NO"]); continue
        cats = [CATS[c] for c in range(5) if pc[i, c] >= thr_cat[c]]
        out.append(cats if cats else [CATS[int(np.argmax(pc[i]))]])
    return out

def pyevall_hard_full(ids, gold, pred):
    with tempfile.TemporaryDirectory() as td:
        pf, gf = os.path.join(td,"p"), os.path.join(td,"g")
        json.dump([{"test_case":TC,"id":str(i),"value":pred[k]} for k,i in enumerate(ids)], open(pf,"w"))
        json.dump([{"test_case":TC,"id":str(i),"value":gold[k]} for k,i in enumerate(ids)], open(gf,"w"))
        rep = PyEvALLEvaluation().evaluate(pf, gf, ["ICM","ICMNorm","FMeasure"],
            **{PyEvALLUtils.PARAM_REPORT:PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
               PyEvALLUtils.PARAM_HIERARCHY:HIER,
               PyEvALLUtils.PARAM_LOG_LEVEL:PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE})
        m = rep.report["metrics"]
        return (m["ICM"]["results"]["average_per_test_case"],
                m["ICMNorm"]["results"]["average_per_test_case"],
                m["FMeasure"]["results"]["test_cases"][0]["average"])
