"""Configuración global del pipeline EXIST 2026 - subtarea 2.1 (memes)."""
import os
import random
import numpy as np
import torch

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
TEAM_NAME = "Ordantis"   # nombre real del equipo

import glob as _glob

ROOT = os.path.dirname(os.path.abspath(__file__))                 # .../Trabajo
_DATOS = os.path.abspath(os.path.join(ROOT, "..", "datos"))


def _find(pattern, base=_DATOS):
    hits = _glob.glob(os.path.join(base, pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No encontrado: {pattern} bajo {base}")
    return hits[0]


# rutas robustas a espacios/underscores en los nombres de carpeta
TRAIN_JSON = _find("**/*Memes*Dataset*/training/EXIST2026_training.json")
TEST_JSON = _find("**/*Memes*Dataset*/test/EXIST2026_test_clean.json")
MEMES_DIR = os.path.dirname(os.path.dirname(TRAIN_JSON))
DATA_ROOT = os.path.dirname(MEMES_DIR)
TRAIN_IMG_DIR = os.path.join(os.path.dirname(TRAIN_JSON), "memes")
TEST_IMG_DIR = os.path.join(os.path.dirname(TEST_JSON), "memes")
FORMAT_VAL_SCRIPT = _find("**/evaluation/exist2025_format_val_V0.2.py")

OUT_DIR = os.path.join(ROOT, f"exist2026_{TEAM_NAME}")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PRE_DIR = os.path.join(OUT_DIR, "preprocessed")
VAL_DIR = os.path.join(OUT_DIR, "validation_results")
ATTN_DIR = os.path.join(OUT_DIR, "attention_weights")
GPU_DIR = os.path.join(OUT_DIR, "gpu_logs")

for _d in (OUT_DIR, CKPT_DIR, PRE_DIR, VAL_DIR, ATTN_DIR, GPU_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Modelos base
# --------------------------------------------------------------------------
TEXT_MODEL = "xlm-roberta-base"
VIT_MODEL = "google/vit-base-patch16-224"

# --------------------------------------------------------------------------
# Dimensiones sensoriales
# --------------------------------------------------------------------------
N_ET, N_HR, N_EEG = 24, 4, 80
N_SENSORS = N_ET + N_HR + N_EEG  # 108

# Emociones Ekman (7-dim, ya en [0,1] -> NO z-score). Orden canónico:
EKMAN_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
N_EMOTIONS = len(EKMAN_ORDER)

# --------------------------------------------------------------------------
# Hiperparámetros — entrenamiento en DOS FASES
# --------------------------------------------------------------------------
MAX_TOKENS = 256
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
GRAD_CLIP = 1.0
GRAD_ACCUM = 2
WARMUP_FRAC = 0.10
POS_WEIGHT = 1.2

# Fase 1: head warm-up (XLM-R + ViT congelados)
PHASE1_EPOCHS = 5
PHASE1_LR = 5e-5
# Fase 2: full fine-tune (XLM-R descongelado; ViT sigue congelado)
PHASE2_EPOCHS = 15
PATIENCE = 4                       # early stopping sobre ICMSoft en fase 2
LR_LOW = 1e-5                      # embeddings + layers 0-6
LR_HIGH = 3e-5                     # layers 7-11 + pooler
LR_HEAD = 1e-4                     # heads + ramas sensoriales + Ekman + Gemini

# (compatibilidad)
N_EPOCHS = PHASE2_EPOCHS

BATCH = {"M1": (24, 64), "M2": (16, 64), "M3_ABCE": (16, 64), "M3_D": (16, 64)}

NUM_WORKERS = 8

# split
SEED = 42
VAL_FRAC = 0.15

# threshold scan
THRESHOLDS = [round(0.30 + 0.01 * i, 2) for i in range(41)]  # 0.30..0.70

# zona dudosa
DOUBT_LO, DOUBT_HI = 0.45, 0.55

# vistas M3 (todas con emociones Ekman como input adicional)
M3_VIEWS = {
    "A": dict(seed=42, text=True, image=True, et=True, hr=True, eeg=True, caption=False, set_pool=True, emotions=True),
    "B": dict(seed=123, text=True, image=True, et=False, hr=False, eeg=True, caption=False, set_pool=True, emotions=True),
    "C": dict(seed=2024, text=True, image=True, et=True, hr=True, eeg=False, caption=False, set_pool=True, emotions=True),
    "D": dict(seed=7, text=True, image=True, et=False, hr=False, eeg=False, caption=False, set_pool=False, emotions=True),
    "E": dict(seed=999, text=True, image=False, et=False, hr=False, eeg=True, caption=True, set_pool=True, emotions=True),
}

# Sanity checks: tras Fase 1 solo comprobación blanda (nunca aborta).
# La puerta de calidad real es un assert tras la época 5 de Fase 2, solo en M1.
STRICT_PHASE2 = os.environ.get("STRICT_PHASE2", "1") == "1"

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
FORCE_RECOMPUTE = os.environ.get("FORCE_RECOMPUTE", "0") == "1"
# torch.compile: requiere triton+gcc funcionando; en este entorno la build de triton falla,
# así que por defecto OFF. Poner USE_COMPILE=1 para intentarlo.
USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def configure_gpu():
    """BF16 + matmul precision + cudnn benchmark + TF32."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        major, _ = torch.cuda.get_device_capability(0)
        print(f"[GPU] BF16 soportado: {torch.cuda.is_bf16_supported()}")
    else:
        print("[GPU] *** No hay CUDA disponible ***")


def best_attn_impl():
    """Devuelve 'flash_attention_2' si está disponible, si no 'sdpa'."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


AMP_DTYPE = torch.bfloat16
