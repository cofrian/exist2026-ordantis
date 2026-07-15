"""Carga y preprocesado de datos EXIST 2026 memes (subtarea 2.1)."""
import json
import os
import re
import warnings

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import config as C

# --------------------------------------------------------------------------
# Texto OCR
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = _URL_RE.sub(" ", text)
    # conservar la palabra tras '#', eliminando solo el '#'
    t = re.sub(r"#(\w+)", r"\1", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# --------------------------------------------------------------------------
# Sensores: orden canónico de features por modalidad
# --------------------------------------------------------------------------
def _modality_feature_order(meme_dict):
    """Extrae los nombres de feature en orden para cada modalidad a partir del
    primer meme que tenga un usuario para esa modalidad."""
    order = {}
    for mod in ("ET", "HR", "EEG"):
        for m in meme_dict.values():
            bu = m["sensorial"]["modalities"][mod]["by_user"]
            if bu:
                order[mod] = list(next(iter(bu.values())).keys())
                break
    return order


def _user_vector(feat_dict, feat_order):
    return np.array([feat_dict.get(k, np.nan) if feat_dict.get(k) is not None else np.nan
                     for k in feat_order], dtype=np.float64)


# --------------------------------------------------------------------------
# Etiquetas
# --------------------------------------------------------------------------
def soft_label(meme):
    labels = meme["labels_task2_1"]
    return sum(1 for x in labels if x == "YES") / len(labels)


# --------------------------------------------------------------------------
# Carga principal
# --------------------------------------------------------------------------
def load_split():
    """Devuelve dicts train/test ya con texto limpio y matrices sensoriales por sujeto.

    Cada ejemplo:
        {
          "id": str, "lang": "es"|"en", "text": str,
          "img_path": str (puede no existir),
          "sensors": {"ET": (n_u, 24), "HR": (n_u, 4), "EEG": (n_u, 80)},  # nan donde falte
          "soft": float|None,
        }
    """
    train_raw = json.load(open(C.TRAIN_JSON, encoding="utf-8"))
    test_raw = json.load(open(C.TEST_JSON, encoding="utf-8"))

    feat_order = _modality_feature_order(train_raw)
    assert len(feat_order["ET"]) == C.N_ET
    assert len(feat_order["HR"]) == C.N_HR
    assert len(feat_order["EEG"]) == C.N_EEG

    def build(raw, img_dir, has_labels):
        out = []
        for k, m in raw.items():
            sens = {}
            for mod in ("ET", "HR", "EEG"):
                bu = m["sensorial"]["modalities"][mod]["by_user"]
                if bu:
                    mat = np.stack([_user_vector(v, feat_order[mod]) for v in bu.values()], axis=0)
                else:
                    mat = np.full((0, len(feat_order[mod])), np.nan)
                sens[mod] = mat
            out.append(dict(
                id=str(m["id_EXIST"]),
                lang=m["lang"],
                text=clean_text(m.get("text", "")),
                img_path=os.path.join(img_dir, os.path.basename(m["meme"])),
                sensors=sens,
                soft=(soft_label(m) if has_labels else None),
            ))
        return out

    train = build(train_raw, C.TRAIN_IMG_DIR, True)
    test = build(test_raw, C.TEST_IMG_DIR, False)

    # emociones Ekman (7-dim, ya en [0,1] -> sin z-score). Si no se han pre-calculado -> ceros.
    emo_path = os.path.join(C.PRE_DIR, "ekman_emotions.json")
    emo = json.load(open(emo_path)) if os.path.exists(emo_path) else {}
    if not emo:
        warnings.warn("ekman_emotions.json no encontrado: se usarán ceros (ejecuta precompute_emotions.py).")
    for e in train + test:
        v = emo.get(e["id"])
        e["emotions"] = (np.asarray(v, dtype=np.float32) if v is not None
                         else np.zeros(C.N_EMOTIONS, dtype=np.float32))

    if C.DRY_RUN:
        # 50 train + 20 val (=> 70 del train) + 20 test
        train = train[:70]
        test = test[:20]

    # --- split estratificado 85/15 por idioma Y por soft binarizada
    strat = [f"{e['lang']}_{int(e['soft'] >= 0.5)}" for e in train]
    tr_idx, va_idx = train_test_split(
        np.arange(len(train)), test_size=C.VAL_FRAC, random_state=C.SEED, stratify=strat)
    tr = [train[i] for i in tr_idx]
    va = [train[i] for i in va_idx]

    # --- estadísticos z-score por modalidad sobre TRAIN (concatenando sujetos)
    stats = compute_sensor_stats(tr, feat_order)
    apply_sensor_norm(tr, stats)
    apply_sensor_norm(va, stats)
    apply_sensor_norm(test, stats)

    return dict(train=tr, val=va, test=test, feat_order=feat_order, sensor_stats=stats)


def drop_ties(examples):
    """Para la ablación 'baseline' (paper Arcos): descarta los empates 3-3 (soft == 0.5)."""
    return [e for e in examples if e["soft"] is None or abs(e["soft"] - 0.5) > 1e-9]


def compute_sensor_stats(examples, feat_order):
    stats = {}
    for mod in ("ET", "HR", "EEG"):
        rows = [e["sensors"][mod] for e in examples if e["sensors"][mod].shape[0] > 0]
        allrows = np.concatenate(rows, axis=0) if rows else np.zeros((1, len(feat_order[mod])))
        mean = np.nanmean(allrows, axis=0)
        std = np.nanstd(allrows, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)
        stats[mod] = dict(mean=mean.tolist(), std=std.tolist())
    return stats


def apply_sensor_norm(examples, stats):
    """Z-score; NaN (feature ausente) -> 0 tras normalizar.
    Si un meme tiene 0 sujetos para una modalidad -> vector medio del train (=> 0 tras z-score)."""
    for e in examples:
        norm = {}
        for mod in ("ET", "HR", "EEG"):
            mean = np.array(stats[mod]["mean"])
            std = np.array(stats[mod]["std"])
            mat = e["sensors"][mod]
            if mat.shape[0] == 0:
                z = np.zeros((1, mean.shape[0]))            # un "sujeto medio"
            else:
                z = (mat - mean) / std
                z = np.where(np.isnan(z), 0.0, z)
            norm[mod] = z.astype(np.float32)
        e["sensors_z"] = norm
        # promedios (para M1 y para la regla de zona dudosa)
        e["sensors_avg"] = {mod: norm[mod].mean(axis=0).astype(np.float32) for mod in norm}


# --------------------------------------------------------------------------
# Imágenes
# --------------------------------------------------------------------------
_PROBLEM_LOG = os.path.join(C.VAL_DIR, "images_problematic.txt")


def load_image(path):
    """Devuelve PIL RGB 'tal cual' (sin resize). Imagen blanca 224x224 si falla."""
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as ex:
        warnings.warn(f"Imagen problemática {path}: {ex}")
        with open(_PROBLEM_LOG, "a") as f:
            f.write(f"{path}\t{ex}\n")
        return Image.new("RGB", (224, 224), (255, 255, 255))


# --------------------------------------------------------------------------
# Helpers PyEvALL: escribir ficheros de predicciones/gold
# --------------------------------------------------------------------------
def write_hard_file(path, ids, labels, test_case="EXIST2025"):
    data = [{"test_case": test_case, "id": str(i), "value": ("YES" if v else "NO")}
            for i, v in zip(ids, labels)]
    json.dump(data, open(path, "w"), ensure_ascii=False)


def write_soft_file(path, ids, probs, test_case="EXIST2025"):
    data = []
    for i, p in zip(ids, probs):
        p = float(min(max(p, 0.0), 1.0))
        yes = round(p, 4)
        no = round(1.0 - yes, 4)
        # normalización exacta a suma 1.0
        s = yes + no
        yes, no = yes / s, no / s
        data.append({"test_case": test_case, "id": str(i), "value": {"YES": yes, "NO": no}})
    json.dump(data, open(path, "w"), ensure_ascii=False)


def write_gold_hard(path, ids, soft_targets, test_case="EXIST2025"):
    data = [{"test_case": test_case, "id": str(i), "value": ("YES" if t >= 0.5 else "NO")}
            for i, t in zip(ids, soft_targets)]
    json.dump(data, open(path, "w"), ensure_ascii=False)


def write_gold_soft(path, ids, soft_targets, test_case="EXIST2025"):
    data = [{"test_case": test_case, "id": str(i),
             "value": {"YES": float(t), "NO": float(1.0 - t)}}
            for i, t in zip(ids, soft_targets)]
    json.dump(data, open(path, "w"), ensure_ascii=False)
