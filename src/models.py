"""Modelos M1 / M2 / M3-vistas para EXIST 2026 subtarea 2.1."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import config as C


def encode_text_mean_pool(xlm_roberta, input_ids, attention_mask):
    """Mean-pooling enmascarado sobre los tokens de XLM-RoBERTa.

    NOTA: desviación intencional del paper de Arcos (que usa el token [CLS]).
    El mean-pooling enmascarado da representaciones congeladas mejores que [CLS]
    (SBERT, Reimers & Gurevych 2019). Aporta valor tanto en Fase 1 (warm-up con
    XLM-R congelado) como en Fase 2 (fine-tune). Se usa en M1/M2/M3 y las 3 subtasks.
    """
    out = xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = out.last_hidden_state                                  # (B, L, 768)
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)            # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)                             # (B, 768)
    counts = mask.sum(dim=1).clamp(min=1e-9)                            # (B, 1)
    return summed / counts                                              # (B, 768)


class SetAttentionPool(nn.Module):
    """Pooling con atención sobre un conjunto de sujetos -> vector fijo (256)."""
    def __init__(self, in_dim, hid=256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(C.DROPOUT),
            nn.Linear(hid, hid), nn.GELU(),
        )
        self.attn = nn.Linear(hid, 1)

    def forward(self, x, mask):
        """x: (B, S, in_dim)  mask: (B, S) bool (True=válido). Devuelve (B, hid) y pesos (B,S)."""
        h = self.phi(x)                                    # (B,S,hid)
        scores = self.attn(h).squeeze(-1)                  # (B,S)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)               # (B,S)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        vec = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B,hid)
        return vec, alpha


class MemeClassifier(nn.Module):
    """Clasificador binario configurable.

    cfg keys: text, image, et, hr, eeg, caption, set_pool (bool).
    """
    def __init__(self, cfg, attn_impl=None):
        super().__init__()
        self.cfg = cfg
        attn_impl = attn_impl or C.best_attn_impl()

        feat_dim = 0
        if cfg["text"]:
            self.text_model = AutoModel.from_pretrained(
                C.TEXT_MODEL, torch_dtype=C.AMP_DTYPE, attn_implementation=attn_impl)
            self.text_model.gradient_checkpointing_enable()
            feat_dim += self.text_model.config.hidden_size  # 768
        if cfg["image"]:
            feat_dim += 768                                  # embedding ViT precalculado

        self.set_pool = cfg.get("set_pool", False)
        self.mods = [m for m in ("ET", "HR", "EEG")
                     if cfg[{"ET": "et", "HR": "hr", "EEG": "eeg"}[m]]]
        self.mod_dims = {"ET": C.N_ET, "HR": C.N_HR, "EEG": C.N_EEG}
        if self.set_pool:
            self.pools = nn.ModuleDict({m: SetAttentionPool(self.mod_dims[m], 256) for m in self.mods})
            feat_dim += 256 * len(self.mods)
        else:
            feat_dim += sum(self.mod_dims[m] for m in self.mods)

        self.use_emotions = cfg.get("emotions", False)
        if self.use_emotions:
            feat_dim += C.N_EMOTIONS   # 7-dim Ekman, ya en [0,1] -> sin z-score

        self.head = nn.Sequential(
            nn.Dropout(C.DROPOUT),
            nn.Linear(feat_dim, 512), nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(512, 1),
        )
        self._feat_dim = feat_dim
        self._debug_first_batch = os.environ.get("DEBUG_FWD", "0") == "1"

    def forward(self, batch, return_attn=False):
        feats = []
        names = []
        if self.cfg["text"]:
            # mean-pooling enmascarado (no [CLS]) — ver encode_text_mean_pool
            text_emb = encode_text_mean_pool(self.text_model, batch["input_ids"],
                                             batch["attention_mask"]).float()
            feats.append(text_emb); names.append("text_meanpool")
        if self.cfg["image"]:
            feats.append(batch["img_emb"].float()); names.append("img_emb")
        attn_out = {}
        for m in self.mods:
            if self.set_pool:
                vec, alpha = self.pools[m](batch[f"sens_{m}"].float(), batch[f"mask_{m}"])
                feats.append(vec); names.append(f"sens_{m}_pool")
                attn_out[m] = alpha
            else:
                feats.append(batch[f"sens_{m}_avg"].float()); names.append(f"sens_{m}_avg")
        if self.use_emotions:
            feats.append(batch["emotions"].float()); names.append("emotions")
        # ===== DEBUG: solo primera batch =====
        if getattr(self, "_debug_first_batch", False):
            print(f"\n{'='*60}\nDEBUG FORWARD — PRIMERA BATCH ({self.cfg})\n{'='*60}")
            for nm, ft in zip(names, feats):
                bad = torch.isnan(ft).any().item() or torch.isinf(ft).any().item()
                print(f"  {nm:16s} shape={tuple(ft.shape)} mean={ft.float().mean().item():+.4f} "
                      f"std={ft.float().std().item():.4f} min={ft.float().min().item():+.3f} "
                      f"max={ft.float().max().item():+.3f} dev={ft.device} {'NAN/INF!!' if bad else ''}")
                print(f"  {'':16s} sample[0,:5]={[round(v,4) for v in ft[0,:5].float().tolist()]}")
            if torch.cuda.is_available():
                print(f"  VRAM allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            print(f"{'='*60}\n")
            self._debug_first_batch = False
        # ===== FIN DEBUG =====
        x = torch.cat(feats, dim=1)
        logit = self.head(x).squeeze(-1)
        if return_attn:
            return logit, attn_out
        return logit

    def param_groups(self):
        """LRs diferenciados: capas bajas (0-6), altas (7-12), heads."""
        groups = []
        if self.cfg["text"]:
            low, high = [], []
            for n, p in self.text_model.named_parameters():
                layer = _layer_index(n)
                (low if layer is not None and layer <= 6 else high).append(p)
            groups.append({"params": low, "lr": C.LR_LOW})
            groups.append({"params": high, "lr": C.LR_HIGH})
        head_params = list(self.head.parameters())
        if self.set_pool:
            head_params += list(self.pools.parameters())
        groups.append({"params": head_params, "lr": C.LR_HEAD})
        return groups


def _layer_index(name):
    # xlm-roberta: encoder.layer.<i>.  embeddings.* -> capa 0
    if "embeddings" in name:
        return 0
    import re
    m = re.search(r"encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1   # layer 0 -> 1 ... layer 11 -> 12
    return None  # pooler u otros -> tratado como "alto"
