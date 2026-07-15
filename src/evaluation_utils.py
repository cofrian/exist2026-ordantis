"""Evaluación con PyEvALL + optimización de threshold + temperature scaling."""
import json
import os
import tempfile

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

import config as C
import data as D


METRICS_HARD = ["ICM", "ICMNorm", "FMeasure"]
METRICS_SOFT = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]


def evaluate_predictions(predictions_path, gold_path, mode="hard"):
    """Wrapper helper obligatorio. Devuelve el dict de resultados de PyEvALL."""
    evaluator = PyEvALLEvaluation()
    params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
              PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE}
    metrics = METRICS_HARD if mode == "hard" else METRICS_SOFT
    report = evaluator.evaluate(predictions_path, gold_path, metrics, **params)
    # versión instalada de PyEvALL: el report embebido expone .report (dict)
    if hasattr(report, "get_results_as_dict"):
        return report.get_results_as_dict()
    return getattr(report, "report", report)


def metric_value(results_dict, metric_name):
    """Extrae el valor (promedio sobre test cases) de una métrica del report de PyEvALL."""
    try:
        m = results_dict["metrics"][metric_name]
        if m.get("status") != "OK":
            return None
        return float(m["results"]["average_per_test_case"])
    except Exception:
        return None


# --------------------------------------------------------------------------
# Evaluación a partir de arrays en memoria (crea ficheros temporales)
# --------------------------------------------------------------------------
def eval_hard(ids, pred_bool, soft_targets):
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pred.json")
        gp = os.path.join(td, "gold.json")
        D.write_hard_file(pp, ids, pred_bool)
        D.write_gold_hard(gp, ids, soft_targets)
        res = evaluate_predictions(pp, gp, "hard")
    return {m: metric_value(res, m) for m in METRICS_HARD}


def eval_soft(ids, probs, soft_targets):
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "pred.json")
        gp = os.path.join(td, "gold.json")
        D.write_soft_file(pp, ids, probs)
        D.write_gold_soft(gp, ids, soft_targets)
        res = evaluate_predictions(pp, gp, "soft")
    return {m: metric_value(res, m) for m in METRICS_SOFT}


# --------------------------------------------------------------------------
# Mejora B: threshold óptimo con ICM real
# --------------------------------------------------------------------------
def optimize_threshold(ids, probs, soft_targets, thresholds=None):
    thresholds = thresholds or C.THRESHOLDS
    best_t, best_icm = 0.5, -1e18
    history = []
    for t in thresholds:
        pred = probs >= t
        icm = eval_hard(ids, pred, soft_targets)["ICM"]
        if icm is None:
            icm = -1e18
        history.append((t, icm))
        if icm > best_icm:
            best_icm, best_t = icm, t
    return best_t, best_icm, history


# --------------------------------------------------------------------------
# Mejora C: temperature scaling para soft
# --------------------------------------------------------------------------
def fit_temperature(logits, soft_targets):
    """Minimiza la BCE (cross-entropy contínua) escalando logits por 1/T."""
    logits = np.asarray(logits, dtype=np.float64)
    t = np.asarray(soft_targets, dtype=np.float64)
    eps = 1e-7

    def nll(T):
        p = np.clip(expit(logits / T), eps, 1 - eps)
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    return float(res.x)


def apply_temperature(logits, T):
    return expit(np.asarray(logits, dtype=np.float64) / T)


# --------------------------------------------------------------------------
# Baselines oficiales
# --------------------------------------------------------------------------
def baseline_metrics(ids, soft_targets):
    n = len(ids)
    maj = eval_hard(ids, np.ones(n, dtype=bool), soft_targets)    # siempre YES
    mino = eval_hard(ids, np.zeros(n, dtype=bool), soft_targets)  # siempre NO
    return dict(baseline_majority=maj, baseline_minority=mino)
