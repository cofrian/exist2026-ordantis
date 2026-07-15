"""Entrenamiento en DOS FASES (head warm-up + full fine-tune) de un modelo
(M1, M2 o una vista de M3) para EXIST 2026.  Subtarea binaria (2.1) por ahora;
las cabezas de 2.2 / 2.3 se enchufan vía cfg['task'] cuando estén listas.
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config as C
import data as D
import evaluation_utils as E
from dataset import MemeDataset, make_collate, to_device
from models import MemeClassifier

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


# --------------------------------------------------------------------------
@torch.no_grad()
def infer_logits(model, loader):
    model.eval()
    ids, logits, targets = [], [], []
    for batch in loader:
        batch = to_device(batch, C.DEVICE)
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(batch)
        ids.extend(batch["id"])
        logits.append(logit.float().cpu().numpy())
        targets.append(batch["soft"].cpu().numpy())
    return ids, np.concatenate(logits), np.concatenate(targets)


def _gpu_peak():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += p.grad.data.norm(2).item() ** 2
    return tot ** 0.5


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def _validate(model, dl_va, name, epoch_tag, train_loss, dt, n_batches, n_seen, gpu_csv, phase):
    ids, logits, targets = infer_logits(model, dl_va)
    probs = 1.0 / (1.0 + np.exp(-logits))
    val_loss = float(np.mean(-(targets * np.log(np.clip(probs, 1e-7, 1)) +
                               (1 - targets) * np.log(np.clip(1 - probs, 1e-7, 1)))))
    if roc_auc_score is not None and len(set((targets >= 0.5).tolist())) > 1:
        auc = float(roc_auc_score((targets >= 0.5).astype(int), probs))
    else:
        auc = float("nan")
    soft_m = E.eval_soft(ids, probs, targets)
    hard_m = E.eval_hard(ids, probs >= 0.5, targets)
    icm_soft = soft_m.get("ICMSoft")
    icm_soft = icm_soft if icm_soft is not None else -1e18
    sps = n_seen / dt if dt > 0 else 0.0
    print(f"  [{name} {epoch_tag}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
          f"val_AUC={auc:.4f} | val_ICMSoft={icm_soft:.4f} | val_ICM={hard_m.get('ICM')} | "
          f"VRAM={_gpu_peak():.1f}GB | {dt/60:.1f}min | {sps:.0f} sps")
    with open(gpu_csv, "a", newline="") as f:
        csv.writer(f).writerow([phase, epoch_tag, round(_gpu_peak(), 2),
                                round(dt / max(1, n_batches), 3), round(sps, 1),
                                round(train_loss, 4), round(val_loss, 4),
                                round(icm_soft, 4), round(auc, 4) if auc == auc else ""])
    return dict(ids=ids, logits=logits, probs=probs, targets=targets,
                auc=auc, icm_soft=icm_soft, hard=hard_m, soft=soft_m, val_loss=val_loss)


def _run_epoch(model, dl_tr, opt, sched, hard_target, pos_weight, track_params=None):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    run_loss, n_seen, n_batches = 0.0, 0, 0
    last_gn = {}
    opt.zero_grad(set_to_none=True)
    for step, batch in enumerate(dl_tr):
        batch = to_device(batch, C.DEVICE)
        tgt = (batch["soft"] >= 0.5).float() if hard_target else batch["soft"]
        with torch.autocast("cuda", dtype=C.AMP_DTYPE, enabled=torch.cuda.is_available()):
            logit = model(batch)
            loss = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pos_weight) / C.GRAD_ACCUM
        loss.backward()
        run_loss += loss.item() * C.GRAD_ACCUM * batch["soft"].size(0)
        n_seen += batch["soft"].size(0)
        n_batches += 1
        if (step + 1) % C.GRAD_ACCUM == 0 or (step + 1) == len(dl_tr):
            if track_params:
                last_gn = {k: _grad_norm(v) for k, v in track_params.items()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
    dt = time.time() - t0
    return run_loss / max(1, n_seen), n_batches, n_seen, dt, last_gn


# --------------------------------------------------------------------------
def check_phase1_health(history_p1, model_name):
    """Comprueba salud de Fase 1 (warm-up del head con XLM-R congelado).

    NO aborta nunca; solo informa. La calidad real se valida tras Fase 2 ep5.
    history_p1: lista de dicts {epoch, train_loss, val_loss, val_auc, val_icm_soft}.
    """
    losses = [ep["train_loss"] for ep in history_p1]
    val_losses = [ep["val_loss"] for ep in history_p1]
    last = history_p1[-1]
    issues = []
    if losses[0] - losses[-1] < 0.005:
        issues.append(f"train_loss plano: {losses[0]:.4f}->{losses[-1]:.4f} (esperado >=0.005 de bajada)")
    for i, ep in enumerate(history_p1):
        for key in ("train_loss", "val_loss", "val_auc", "val_icm_soft"):
            v = ep.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                issues.append(f"NaN/None en {key} de época {i+1}")
    if last["val_auc"] == last["val_auc"] and last["val_auc"] < 0.55:
        issues.append(f"val_AUC={last['val_auc']:.4f} < 0.55 (peor que aleatorio)")
    if val_losses[-1] > val_losses[0] + 0.05:
        issues.append(f"val_loss explotando: {val_losses[0]:.4f}->{val_losses[-1]:.4f}")
    if issues:
        print(f"⚠️  [{model_name}] Fase 1 con avisos (continuamos a Fase 2):")
        for it in issues:
            print(f"     - {it}")
        return False
    print(f"✅ [{model_name}] Fase 1 saludable: train_loss {losses[0]:.4f}->{losses[-1]:.4f}, "
          f"val_AUC final={last['val_auc']:.4f}, sin NaN")
    return True


# --------------------------------------------------------------------------
def train_model(name, cfg, batch_train, batch_infer, seed, use_caption=False,
                splits=None, vit_emb=None, captions=None, hard_target=False,
                strict_phase1=False, strict_phase2=False):
    """Entrena en dos fases y devuelve dict con model, val_ids, val_logits, val_targets, best_metrics."""
    print(f"\n{'='*60}\n{name}  (seed={seed}, hard_target={hard_target}, cfg={cfg})\n{'='*60}")
    C.set_seed(seed)

    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    collate = make_collate(tok)
    ds_tr = MemeDataset(splits["train"], tok, vit_emb, captions, use_caption)
    ds_va = MemeDataset(splits["val"], tok, vit_emb, captions, use_caption)

    nw = 0 if C.DRY_RUN else C.NUM_WORKERS
    dl_kwargs = dict(num_workers=nw, pin_memory=True, collate_fn=collate)
    if nw > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    dl_tr = DataLoader(ds_tr, batch_size=batch_train, shuffle=True, drop_last=False, **dl_kwargs)
    dl_va = DataLoader(ds_va, batch_size=batch_infer, shuffle=False, **dl_kwargs)

    model = MemeClassifier(cfg).to(C.DEVICE)
    pos_weight = torch.tensor([C.POS_WEIGHT], device=C.DEVICE)

    gpu_csv = os.path.join(C.GPU_DIR, f"{name}_gpu_stats.csv")
    with open(gpu_csv, "w", newline="") as f:
        csv.writer(f).writerow(["phase", "epoch", "vram_gb", "batch_time_s", "samples_per_sec",
                                "loss_train", "loss_val", "icm_soft_val", "auc_val"])

    p1_epochs = 1 if C.DRY_RUN else C.PHASE1_EPOCHS
    p2_epochs = 2 if C.DRY_RUN else C.PHASE2_EPOCHS

    def steps(n_ep):
        return max(1, (len(dl_tr) // C.GRAD_ACCUM) * n_ep)

    # ================= FASE 1 — head warm-up (XLM-R + ViT congelados) =================
    print(f"[{name}] FASE 1: head warm-up ({p1_epochs} épocas, lr={C.PHASE1_LR})")
    _set_requires_grad(model.text_model, False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  parámetros entrenables: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=C.PHASE1_LR, weight_decay=C.WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP_FRAC * steps(p1_epochs)), steps(p1_epochs))
    history_p1 = []
    for ep in range(1, p1_epochs + 1):
        tl, nb, ns, dt, _ = _run_epoch(model, dl_tr, opt, sched, hard_target, pos_weight)
        val = _validate(model, dl_va, name, f"F1 {ep}/{p1_epochs}", tl, dt, nb, ns, gpu_csv, 1)
        history_p1.append(dict(epoch=ep, train_loss=tl, val_loss=val["val_loss"],
                               val_auc=val["auc"], val_icm_soft=val["icm_soft"]))
    # Comprobación blanda de Fase 1 (no aborta nunca). La puerta de calidad real
    # está en Fase 2 ep5 con strict_phase2. (strict_phase1 queda como flag legacy ignorada)
    if history_p1:
        check_phase1_health(history_p1, name)
    del opt, sched
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================= FASE 2 — full fine-tune (XLM-R descongelado) =================
    print(f"[{name}] FASE 2: full fine-tune ({p2_epochs} épocas, lr low/high/head={C.LR_LOW}/{C.LR_HIGH}/{C.LR_HEAD})")
    _set_requires_grad(model.text_model, True)
    low, high = [], []
    for n_, p in model.text_model.named_parameters():
        if "embeddings" in n_:
            low.append(p)
        elif "encoder.layer." in n_:
            ln = int(n_.split("encoder.layer.")[1].split(".")[0])
            (low if ln <= 6 else high).append(p)
        else:
            high.append(p)  # pooler u otros
    head = [p for n_, p in model.named_parameters() if not n_.startswith("text_model.")]
    opt = torch.optim.AdamW([
        {"params": low, "lr": C.LR_LOW},
        {"params": high, "lr": C.LR_HIGH},
        {"params": head, "lr": C.LR_HEAD},
    ], weight_decay=C.WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP_FRAC * steps(p2_epochs)), steps(p2_epochs))

    best_icm_soft = -1e18
    best_state, best_metrics = None, {}
    patience = 0
    track = {"low": low, "high": high, "head": head}
    for ep in range(1, p2_epochs + 1):
        tl, nb, ns, dt, gn = _run_epoch(model, dl_tr, opt, sched, hard_target, pos_weight, track_params=track)
        val = _validate(model, dl_va, name, f"F2 {ep}/{p2_epochs}", tl, dt, nb, ns, gpu_csv, 2)
        print(f"     grad: low={gn.get('low', 0):.3f} | high={gn.get('high', 0):.3f} | head={gn.get('head', 0):.3f}")
        if not C.DRY_RUN:
            if gn.get("low", 1) < 0.01 or gn.get("high", 1) < 0.01:
                print(f"     ⚠️ aviso: grad_norms muy bajos (low={gn.get('low', 0):.4f}, "
                      f"high={gn.get('high', 0):.4f}). Posible LR insuficiente.")
            if ep == 5:
                # Guardián de calidad real: solo M1, solo si strict_phase2.
                if name == "M1" and strict_phase2:
                    if val["auc"] < 0.70:
                        raise AssertionError(
                            f"⚠️ M1 Fase 2 época 5 no converge: val_AUC={val['auc']:.4f} < 0.70. "
                            f"Detén antes de gastar más GPU en M2/M3. Revisar LRs, sampler, loss o datos.")
                    print(f"     ✅ M1 Fase 2 ep5: val_AUC={val['auc']:.4f} >= 0.70. Modelo aprende.")
                elif val["auc"] < 0.72:
                    print(f"     ⚠️ aviso: {name} Fase 2 época 5 val_AUC={val['auc']:.4f} < 0.72")
        if val["icm_soft"] > best_icm_soft:
            best_icm_soft = val["icm_soft"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(epoch=ep, val_icm_soft=val["icm_soft"], val_icm=val["hard"].get("ICM"),
                                val_f1=val["hard"].get("FMeasure"),
                                val_auc=val["auc"] if val["auc"] == val["auc"] else None)
            patience = 0
            print(f"     mejor ICMSoft={val['icm_soft']:.4f} → checkpoint")
        else:
            patience += 1
            print(f"     early stopping {patience}/{C.PATIENCE}")
            if patience >= C.PATIENCE:
                print("     🛑 early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = os.path.join(C.CKPT_DIR, f"{name}_best.pt")
    torch.save(dict(model_state_dict=best_state, epoch=best_metrics.get("epoch"),
                    val_icm_soft=best_metrics.get("val_icm_soft"), val_icm=best_metrics.get("val_icm"),
                    val_f1=best_metrics.get("val_f1"), val_auc=best_metrics.get("val_auc"), config=cfg),
               ckpt_path)
    print(f"[{name}] checkpoint -> {ckpt_path}")

    if name == "M1":
        bva = best_metrics.get("val_auc")
        if bva is None:
            print("ℹ️  M1 final: sin val_AUC registrado (¿0 épocas de Fase 2 en debug?).")
        elif bva < 0.72:
            print(f"⚠️  M1 final: best val_AUC={bva:.4f} < 0.72. Por debajo del paper. "
                  f"M2/M3 pueden salvar la corrida, pero atento.")
        elif bva < 0.75:
            print(f"ℹ️  M1 final: best val_AUC={bva:.4f}. Razonable. M2/M3 deberían mejorar.")
        else:
            print(f"✅ M1 final: best val_AUC={bva:.4f}. Buen indicador. Continuamos con M2 y M3.")

    val_ids, val_logits, val_targets = infer_logits(model, dl_va)
    model.cpu()
    del opt, sched
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(model=model, tokenizer=tok, collate=collate, vit_emb=vit_emb,
                cfg=cfg, use_caption=use_caption,
                val_ids=val_ids, val_logits=val_logits, val_targets=val_targets,
                best_metrics=best_metrics)
