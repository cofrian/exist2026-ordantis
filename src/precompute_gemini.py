"""Pre-cálculo OFFLINE con Gemini (async, concurrente) — UNA llamada por meme que cubre las 3 subtasks.

SDK nuevo: `google-genai` (`from google import genai`), API asíncrona `client.aio`.

Concurrencia: asyncio.Semaphore(GEMINI_CONCURRENCY=15 por defecto).
Robustez:  exponential backoff (1,2,4,8,16 s) ante 429 / 5xx;  jitter aleatorio entre peticiones.
Modelo por defecto: gemini-3-flash-preview  (override con env GEMINI_MODEL).

Necesita un fichero .env (en la raíz del repo) con  GEMINI_API_KEY=...
Si no hay key o no está la librería, sale con código 0 (NO fatal): el pipeline sigue sin features Gemini.
Salida (caché incremental): exist2026_Ordantis/preprocessed/gemini_predictions.json   (dict id -> dict|null)
Al terminar escribe el marcador:   exist2026_Ordantis/preprocessed/gemini_predictions.DONE

La `description` + `sexism_analysis` que devuelve Gemini se usan además como "caption" de la
Vista E de M3 (ver run_full.load_gemini_captions).
"""
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import config as C

CACHE = Path(C.PRE_DIR) / "gemini_predictions.json"
DONE_MARKER = Path(C.PRE_DIR) / "gemini_predictions.DONE"
DEFAULT_MODEL = "gemini-3-flash-preview"
CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "15"))
SAVE_EVERY = int(os.getenv("GEMINI_SAVE_EVERY", "25"))
MAX_RETRIES = 6  # backoff: 1,2,4,8,16,32 s

GEMINI_PROMPT = """You are an expert annotator analyzing a meme for the EXIST 2026 shared task at CLEF,
which focuses on automatic detection of sexism in social media memes.

You will analyze this meme according to THREE complementary tasks:

TASK 1 - Binary sexism detection
A meme is SEXIST if it expresses, describes, perpetuates or criticizes sexist
behavior, stereotypes or discrimination against women.

TASK 2 - Author intention (only relevant if sexist)
- DIRECT: the author endorses or perpetuates sexism. Sexist content presented as message.
- JUDGEMENTAL: the author criticizes or denounces sexism. Uses irony, sarcasm, or contrast
  to expose sexist behavior. Look for contradictions between image and text, mocking tone,
  hashtags used ironically, showing absurdity of sexist beliefs.

TASK 3 - Categories of sexism (multi-label, can have several or none)
- IDEOLOGICAL-INEQUALITY: denies inequality, discredits feminism, claims men are oppressed.
- STEREOTYPING-DOMINANCE: women in submissive roles, "women's place is kitchen",
  weak/emotional women, men as dominant.
- OBJECTIFICATION: women reduced to body parts, focus on physical attributes only, depersonalization.
- SEXUAL-VIOLENCE: sexual harassment, assault, rape culture, unwanted sexual advances.
- MISOGYNY-NON-SEXUAL-VIOLENCE: hatred toward women, non-sexual aggression, gendered insults.

OCR text extracted from the meme: "{ocr_text}"

INSTRUCTIONS:
1. Examine image and text carefully.
2. Pay attention to context, irony, sarcasm - especially contradictions between visual and textual elements.
3. A meme can have multiple categories simultaneously (Task 3).
4. If the meme is NOT sexist, set task2_2.intention to "NO" and task2_3.categories_present to [].

Return ONLY a valid JSON object:
{{
  "description": "<brief literal description of image and text>",
  "sexism_analysis": "<analysis of why or why not sexist>",
  "reasoning": "<step-by-step reasoning>",
  "task2_1": {{"sexist_probability": <float 0.0-1.0>, "confidence": <float 0.0-1.0>}},
  "task2_2": {{
    "intention": "<NO | DIRECT | JUDGEMENTAL>",
    "intention_probabilities": {{"NO": <float>, "DIRECT": <float>, "JUDGEMENTAL": <float>}},
    "intention_reasoning": "<why this intention>",
    "irony_detected": <true | false>, "irony_confidence": <float 0.0-1.0>
  }},
  "task2_3": {{
    "categories_present": [<list of categories that apply, can be empty>],
    "category_probabilities": {{
      "IDEOLOGICAL-INEQUALITY": <float>, "STEREOTYPING-DOMINANCE": <float>,
      "OBJECTIFICATION": <float>, "SEXUAL-VIOLENCE": <float>,
      "MISOGYNY-NON-SEXUAL-VIOLENCE": <float>
    }},
    "category_reasoning": "<justification for each category>"
  }}
}}
intention_probabilities must sum to 1.0. Category probabilities are independent (each in 0.0-1.0)."""


def _load_key():
    """Load GEMINI_API_KEY from the environment or a .env file.

    Search order: (1) an already-exported env var, (2) a path given in
    GEMINI_ENV_FILE, (3) a .env next to the repo root or the source dir.
    No key is ever hard-coded in this repository.
    """
    candidates = []
    if os.getenv("GEMINI_ENV_FILE"):
        candidates.append(Path(os.getenv("GEMINI_ENV_FILE")))
    here = Path(__file__).resolve().parent
    candidates += [here / ".env", here.parent / ".env", here.parent.parent / ".env"]
    for env_path in candidates:
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except Exception:
                for line in env_path.read_text().splitlines():
                    if line.strip() and not line.strip().startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break
    return os.getenv("GEMINI_API_KEY")


def _is_rate_limit(ex):
    s = str(ex)
    return ("429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()
            or "rate" in s.lower() or "503" in s or "UNAVAILABLE" in s or "500" in s)


async def main_async():
    key = _load_key()
    if not key:
        print("⚠️ GEMINI_API_KEY no encontrada (crea un .env en la raíz del repo). Se omite Gemini.")
        sys.exit(0)
    try:
        from google import genai
        from google.genai import types
    except Exception as ex:
        print(f"⚠️ google-genai no instalado ({ex}). Se omite Gemini. (pip install google-genai python-dotenv)")
        sys.exit(0)
    from PIL import Image

    client = genai.Client(api_key=key)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    safety = [types.SafetySetting(category=c, threshold="BLOCK_NONE")
              for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    gen_config = types.GenerateContentConfig(response_mime_type="application/json", safety_settings=safety)

    train = json.load(open(C.TRAIN_JSON, encoding="utf-8"))
    test = json.load(open(C.TEST_JSON, encoding="utf-8"))
    memes = {**train, **test}
    img_dirs = {"train": C.TRAIN_IMG_DIR, "test": C.TEST_IMG_DIR}

    results = json.load(open(CACHE)) if CACHE.exists() else {}
    pending = [mid for mid in memes if not (mid in results and results[mid] is not None)]
    print(f"Gemini ({model_name}, conc={CONCURRENCY}): {len(results)} en caché, "
          f"{len(pending)} pendientes de {len(memes)}", flush=True)
    if not pending:
        DONE_MARKER.write_text(str(len([v for v in results.values() if v])))
        print("✅ Nada pendiente.", flush=True)
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    state = {"done": 0, "ok": sum(1 for v in results.values() if v), "t0": time.time()}
    save_lock = asyncio.Lock()

    def _img_path(info):
        base = os.path.basename(info["meme"])
        for d in img_dirs.values():
            p = os.path.join(d, base)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(base)

    async def _save():
        async with save_lock:
            json.dump(results, open(CACHE, "w"), indent=2)

    async def worker(mid):
        info = memes[mid]
        async with sem:
            await asyncio.sleep(random.uniform(0.02, 0.20))  # jitter
            delay = 1.0
            for attempt in range(MAX_RETRIES):
                try:
                    img = Image.open(_img_path(info))
                    prompt = GEMINI_PROMPT.format(ocr_text=info.get("text", ""))
                    resp = await client.aio.models.generate_content(
                        model=model_name, contents=[prompt, img], config=gen_config)
                    results[mid] = json.loads(resp.text)
                    state["ok"] += 1
                    break
                except Exception as ex:
                    if _is_rate_limit(ex) and attempt < MAX_RETRIES - 1:
                        wait = delay + random.uniform(0, delay * 0.3)  # backoff + jitter
                        print(f"  [{mid}] 429/5xx -> backoff {wait:.1f}s (intento {attempt+1})", flush=True)
                        await asyncio.sleep(wait)
                        delay *= 2
                        continue
                    print(f"  error en {mid}: {str(ex)[:160]}", flush=True)
                    results.setdefault(mid, None)
                    break
        state["done"] += 1
        if state["done"] % 25 == 0:
            sps = state["done"] / max(1e-9, time.time() - state["t0"])
            eta = (len(pending) - state["done"]) / max(1e-9, sps) / 60
            print(f"  {state['done']}/{len(pending)}  ({state['ok']} válidas)  "
                  f"{sps*60:.0f}/min  ETA~{eta:.0f}min", flush=True)
        if state["done"] % SAVE_EVERY == 0:
            await _save()

    await asyncio.gather(*(worker(mid) for mid in pending))
    await _save()
    valid = len([v for v in results.values() if v is not None])
    DONE_MARKER.write_text(str(valid))
    dt = (time.time() - state["t0"]) / 60
    print(f"✅ Gemini completado: {valid}/{len(memes)} memes con datos en {dt:.1f} min", flush=True)


def main():
    try:
        asyncio.run(main_async())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n⚠️ interrumpido — el progreso guardado se conserva (caché incremental).")


if __name__ == "__main__":
    main()
