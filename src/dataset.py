"""Dataset y collate para los modelos de memes."""
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


class MemeDataset(Dataset):
    def __init__(self, examples, tokenizer, vit_emb, captions=None, use_caption=False):
        self.ex = examples
        self.tok = tokenizer
        self.vit_emb = vit_emb
        self.captions = captions or {}
        self.use_caption = use_caption

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        text = e["text"]
        if self.use_caption:
            cap = self.captions.get(e["id"], "")
            text = (cap + " " + text).strip() if cap else text
        item = dict(
            id=e["id"],
            text=text,
            img_emb=torch.from_numpy(np.asarray(self.vit_emb[e["id"]], dtype=np.float32)),
            emotions=torch.from_numpy(np.asarray(e.get("emotions", np.zeros(C.N_EMOTIONS)),
                                                 dtype=np.float32)),
            soft=float(e["soft"]) if e["soft"] is not None else -1.0,
        )
        for m in ("ET", "HR", "EEG"):
            item[f"sens_{m}"] = torch.from_numpy(e["sensors_z"][m])          # (n_u, d)
            item[f"sens_{m}_avg"] = torch.from_numpy(e["sensors_avg"][m])    # (d,)
        return item


def make_collate(tokenizer):
    def collate(batch):
        ids = [b["id"] for b in batch]
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=C.MAX_TOKENS,
                        return_tensors="pt")
        out = dict(
            id=ids,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            img_emb=torch.stack([b["img_emb"] for b in batch]),
            emotions=torch.stack([b["emotions"] for b in batch]),
            soft=torch.tensor([b["soft"] for b in batch], dtype=torch.float32),
        )
        for m in ("ET", "HR", "EEG"):
            mats = [b[f"sens_{m}"] for b in batch]
            S = max(x.shape[0] for x in mats)
            d = mats[0].shape[1]
            padded = torch.zeros(len(batch), S, d, dtype=torch.float32)
            mask = torch.zeros(len(batch), S, dtype=torch.bool)
            for j, x in enumerate(mats):
                n = x.shape[0]
                padded[j, :n] = x
                mask[j, :n] = True
            out[f"sens_{m}"] = padded
            out[f"mask_{m}"] = mask
            out[f"sens_{m}_avg"] = torch.stack([b[f"sens_{m}_avg"] for b in batch])
        return out
    return collate


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}
