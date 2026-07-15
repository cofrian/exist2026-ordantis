"""Pre-cálculo offline cacheable: embeddings ViT (congelado) de los 5037 memes."""
import os
import numpy as np
import torch
from PIL import Image

import config as C
import data as D

VIT_EMB_PATH = os.path.join(C.PRE_DIR, "vit_embeddings.npz")
TEXT_CLEAN_PATH = os.path.join(C.PRE_DIR, "text_clean.json")
SENSOR_STATS_PATH = os.path.join(C.PRE_DIR, "sensor_stats.json")


@torch.no_grad()
def precompute_vit_embeddings(all_examples, force=False):
    """all_examples: lista de dicts con 'id' e 'img_path'. Devuelve dict id->np.array(768)."""
    if os.path.exists(VIT_EMB_PATH) and not force and not C.FORCE_RECOMPUTE:
        z = np.load(VIT_EMB_PATH, allow_pickle=True)
        emb = {str(k): z[k] for k in z.files}
        # comprobar cobertura
        if all(e["id"] in emb for e in all_examples):
            print(f"[ViT] cargados {len(emb)} embeddings cacheados")
            return emb
        print("[ViT] caché incompleta -> recomputando")

    from transformers import AutoImageProcessor, ViTModel
    print(f"[ViT] cargando {C.VIT_MODEL} ...")
    proc = AutoImageProcessor.from_pretrained(C.VIT_MODEL)
    model = ViTModel.from_pretrained(
        C.VIT_MODEL, torch_dtype=C.AMP_DTYPE, attn_implementation=C.best_attn_impl()
    ).to(C.DEVICE).eval()

    emb = {}
    bs = 64
    ids = [e["id"] for e in all_examples]
    paths = [e["img_path"] for e in all_examples]
    for i in range(0, len(ids), bs):
        batch_imgs = [D.load_image(p) for p in paths[i:i + bs]]
        inputs = proc(images=batch_imgs, return_tensors="pt").to(C.DEVICE)
        inputs = {k: v.to(C.AMP_DTYPE) if v.is_floating_point() else v for k, v in inputs.items()}
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        for j, idx in enumerate(ids[i:i + bs]):
            emb[idx] = cls[j]
        if (i // bs) % 10 == 0:
            print(f"[ViT] {i+len(batch_imgs)}/{len(ids)}")

    np.savez_compressed(VIT_EMB_PATH, **{k: v for k, v in emb.items()})
    print(f"[ViT] guardados {len(emb)} embeddings en {VIT_EMB_PATH}")

    del model
    torch.cuda.empty_cache()
    return emb
