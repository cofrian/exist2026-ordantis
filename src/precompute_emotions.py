"""Pre-cálculo OFFLINE de emociones Ekman (7-dim, en [0,1]) sobre el texto OCR.

ES: daveni/twitter-xlm-roberta-emotion-es  (anger,disgust,fear,joy,sadness,surprise,others)
EN: j-hartmann/emotion-english-distilroberta-base (anger,disgust,fear,joy,neutral,sadness,surprise)

Orden canónico de salida (config.EKMAN_ORDER):
    [anger, disgust, fear, joy, neutral, sadness, surprise]
'others' (modelo ES) se mapea a la posición 'neutral'.

Ejecutar antes de run_full.py:   python precompute_emotions.py
Salida: exist2026_<team>/preprocessed/ekman_emotions.json  (dict id -> [7 floats])
"""
import json
import os

import numpy as np
import torch

import config as C
import data as D

OUT = os.path.join(C.PRE_DIR, "ekman_emotions.json")

ES_MODEL = "daveni/twitter-xlm-roberta-emotion-es"
EN_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def _label_to_canon(label):
    l = label.lower().strip()
    if l in ("others", "other", "neutral", "neutro"):
        return "neutral"
    return l


@torch.no_grad()
def _run_model(model_id, texts):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(C.DEVICE).eval()
    id2label = model.config.id2label
    # índice de cada logit -> posición en EKMAN_ORDER
    perm = []
    for i in range(len(id2label)):
        canon = _label_to_canon(id2label[i])
        perm.append(C.EKMAN_ORDER.index(canon) if canon in C.EKMAN_ORDER else None)
    out = []
    bs = 64
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(C.DEVICE)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for row in probs:
            v = np.zeros(C.N_EMOTIONS, dtype=np.float32)
            for j, p in enumerate(row):
                if perm[j] is not None:
                    v[perm[j]] += float(p)
            out.append(v)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    if os.path.exists(OUT) and not C.FORCE_RECOMPUTE:
        print(f"[Ekman] ya existe {OUT} — nada que hacer (FORCE_RECOMPUTE=1 para regenerar)")
        return
    splits = D.load_split()
    all_ex = splits["train"] + splits["val"] + splits["test"]
    es = [e for e in all_ex if e["lang"] == "es"]
    en = [e for e in all_ex if e["lang"] != "es"]
    emo = {}
    if es:
        print(f"[Ekman] {ES_MODEL}  ({len(es)} textos ES) ...")
        for e, v in zip(es, _run_model(ES_MODEL, [x["text"] for x in es])):
            emo[e["id"]] = v.tolist()
    if en:
        print(f"[Ekman] {EN_MODEL}  ({len(en)} textos EN) ...")
        for e, v in zip(en, _run_model(EN_MODEL, [x["text"] for x in en])):
            emo[e["id"]] = v.tolist()
    json.dump(emo, open(OUT, "w"))
    print(f"[Ekman] guardados {len(emo)} vectores en {OUT}")


if __name__ == "__main__":
    main()
