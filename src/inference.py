"""Inferencia sobre test con TTA (Mejora D) + reglas auxiliares (Mejora E/F)."""
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

import config as C
import data as D

# embeddings ViT del test con 5 augmentaciones (TTA). Imagen sin flip horizontal.
TTA_EMB_PATH = os.path.join(C.PRE_DIR, "vit_embeddings_test_tta.npz")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _tta_transforms():
    base_norm = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
    return [
        # pasada 0: resize + center crop (la "estándar")
        T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                   T.CenterCrop(224), T.ToTensor(), base_norm]),
        # pasadas 1-2: random crop 232->224
        T.Compose([T.Resize(232, interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomCrop(224), T.ToTensor(), base_norm]),
        T.Compose([T.Resize(232, interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomCrop(224), T.ToTensor(), base_norm]),
        # pasadas 3-4: color jitter ±0.05  (NO flip horizontal: rompe el texto)
        T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                   T.ColorJitter(brightness=0.05, contrast=0.05), T.ToTensor(), base_norm]),
        T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                   T.ColorJitter(brightness=0.05, contrast=0.05), T.ToTensor(), base_norm]),
    ]


@torch.no_grad()
def precompute_tta_embeddings(examples, cache_path, tag="ViT-TTA", force=False):
    """Devuelve dict id -> np.array (5, 768) con las 5 pasadas TTA. Cacheable."""
    if os.path.exists(cache_path) and not force and not C.FORCE_RECOMPUTE:
        z = np.load(cache_path)
        emb = {str(k): z[k] for k in z.files}
        if all(e["id"] in emb for e in examples):
            print(f"[{tag}] cargados {len(emb)} embeddings cacheados")
            return emb

    from transformers import ViTModel
    print(f"[{tag}] generando embeddings con 5 augmentaciones ({len(examples)} imgs) ...")
    C.set_seed(C.SEED)
    model = ViTModel.from_pretrained(
        C.VIT_MODEL, torch_dtype=C.AMP_DTYPE, attn_implementation=C.best_attn_impl()
    ).to(C.DEVICE).eval()
    transforms = _tta_transforms()
    emb = {}
    bs = 64
    for ti, tf in enumerate(transforms):
        ids = [e["id"] for e in examples]
        for i in range(0, len(ids), bs):
            imgs = [D.load_image(e["img_path"]) for e in examples[i:i + bs]]
            x = torch.stack([tf(im) for im in imgs]).to(C.DEVICE, dtype=C.AMP_DTYPE)
            out = model(pixel_values=x)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
            for j, idx in enumerate(ids[i:i + bs]):
                emb.setdefault(idx, np.zeros((5, 768), dtype=np.float32))
                emb[idx][ti] = cls[j]
        print(f"[{tag}] pasada {ti+1}/5 hecha")
    np.savez_compressed(cache_path, **emb)
    del model
    torch.cuda.empty_cache()
    return emb


@torch.no_grad()
def precompute_test_tta_embeddings(test_examples, force=False):
    """Devuelve dict id -> np.array (5, 768)."""
    return precompute_tta_embeddings(test_examples, TTA_EMB_PATH, "ViT-TTA", force)




@torch.no_grad()
def predict_test(bundle, test_examples, tta_emb, captions=None):
    """Predice probabilidades sobre test promediando las 5 pasadas TTA.
    Devuelve dict: ids, probs (calibradas si bundle trae 'T'), logits_mean, attn (id->{mod:array})."""
    from dataset import MemeDataset, make_collate, to_device
    model = bundle["model"].to(C.DEVICE).eval()
    _moved = True
    tok = bundle["tokenizer"]
    collate = make_collate(tok)
    use_caption = bundle.get("use_caption", False)

    # un dataset "base" cuyo vit_emb sustituiremos por cada pasada
    all_logits = []
    ids_ref = None
    attn_acc = {}
    for pass_idx in range(tta_emb[test_examples[0]["id"]].shape[0]):
        vit_pass = {e["id"]: tta_emb[e["id"]][pass_idx] for e in test_examples}
        ds = MemeDataset(test_examples, tok, vit_pass, captions, use_caption)
        dl = torch.utils.data.DataLoader(ds, batch_size=48, shuffle=False, collate_fn=collate)
        logits, ids = [], []
        for batch in dl:
            batch = to_device(batch, C.DEVICE)
            with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
                want_attn = bundle["cfg"].get("set_pool", False) and pass_idx == 0
                if want_attn:
                    logit, attn = model(batch, return_attn=True)
                    for bi, _id in enumerate(batch["id"]):
                        attn_acc[_id] = {m: attn[m][bi].float().cpu().numpy() for m in attn}
                else:
                    logit = model(batch)
            logits.append(logit.float().cpu().numpy())
            ids.extend(batch["id"])
        all_logits.append(np.concatenate(logits))
        ids_ref = ids

    logits_mean = np.mean(np.stack(all_logits, axis=0), axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits_mean))
    if bundle.get("T"):
        from evaluation_utils import apply_temperature
        probs = apply_temperature(logits_mean, bundle["T"])
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(ids=ids_ref, probs=probs, logits_mean=logits_mean, attn=attn_acc)


# --------------------------------------------------------------------------
# Mejora E — regla auxiliar en zona dudosa
# --------------------------------------------------------------------------
def _rt_index():
    return 0  # 'reaction_time' es la primera feature ET


def _frontal_alpha_indices():
    # EEG: 16 canales x 5 bandas (Delta,Theta,Alpha,Beta,Gamma). Alpha = offset 2.
    # "frontal" -> primeros 4 canales (0-3)
    return [ch * 5 + 2 for ch in range(4)]


def doubt_zone_rule(examples, probs, train_examples):
    """Devuelve array bool de predicciones ajustadas (a aplicar tras el threshold).
    Solo modifica memes con prob en [0.45, 0.55]."""
    # percentil 75 de reaction_time (z-score) en train
    rts_train = []
    for e in train_examples:
        et = e["sensors_z"]["ET"]
        if et.shape[0] > 0:
            rts_train.append(np.nanmean(et[:, _rt_index()]))
    rt_p75 = np.nanpercentile(rts_train, 75) if rts_train else 0.0
    fa = _frontal_alpha_indices()
    adjust = np.zeros(len(examples), dtype=bool)  # True -> forzar YES
    for i, e in enumerate(examples):
        if not (C.DOUBT_LO <= probs[i] <= C.DOUBT_HI):
            continue
        et = e["sensors_z"]["ET"]
        eeg = e["sensors_z"]["EEG"]
        rt_high = et.shape[0] > 0 and np.nanmean(et[:, _rt_index()]) > rt_p75
        alpha_drop = eeg.shape[0] > 0 and np.nanmean(eeg[:, fa]) < -0.5  # por debajo de la media
        if rt_high and alpha_drop:
            adjust[i] = True
    return adjust


def apply_hard_decision(examples, probs, threshold, use_doubt_rule=False, train_examples=None):
    pred = probs >= threshold
    if use_doubt_rule:
        adj = doubt_zone_rule(examples, probs, train_examples)
        pred = pred | adj
    return pred
