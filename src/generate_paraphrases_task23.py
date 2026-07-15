"""Genera paráfrasis (data augmentation) del análisis de Gemini para las clases minoritarias
de Task 2.3 (SEXUAL-VIOLENCE y MISOGYNY-NON-SEXUAL-VIOLENCE).

Sólo parafrasea el TEXTO enriquecido (description / sexism_analysis / reasoning / category_reasoning);
NO toca imágenes ni sensores. No re-promptea Gemini para añadir campos nuevos al JSON original.

SDK: google-genai (`from google import genai`), API asíncrona.  Necesita un .env (raíz del repo) con GEMINI_API_KEY.
Salida: exist2026_Ordantis/preprocessed/task23_paraphrases.json   (dict  meme_id -> [ {description, sexism_analysis, reasoning, category_reasoning}, ... ] )
Si no hay key / librería: sale 0 (no fatal).
"""
import asyncio, json, os, random, sys, time
from pathlib import Path
import config as C

CATS = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
        "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]
GEMINI_PRED = Path(C.PRE_DIR) / "gemini_predictions.json"
OUT = Path(C.PRE_DIR) / "task23_paraphrases.json"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "10"))
MAX_RETRIES = 6
N_MISOGYNY, N_SEXUAL = 10, 5

PROMPT = """You are an expert in discourse analysis on sexism.

Below is the analysis of a sexist meme that belongs to these categories: {categories}.

Task: produce {n} paraphrases of the analysis WITHOUT changing its meaning or the categories.
- Keep the identified sexist categories intact.
- Vary wording, structure and metaphors.
- Keep the same level of detail.
- Do NOT change key facts of the visual description.

Original analysis:
DESCRIPTION: {description}
SEXISM ANALYSIS: {sexism_analysis}
GENERAL REASONING: {reasoning}
CATEGORY REASONING: {category_reasoning}

Return strict JSON:
{{"paraphrases": [{{"description": "...", "sexism_analysis": "...", "reasoning": "...", "category_reasoning": "..."}}, ... ({n} total)]}}
"""


def _ann_cats(entry):
    if isinstance(entry, list):
        return [x for x in entry if x in CATS]
    if entry == "UNKNOWN":
        return None
    return []


def _load_key():
    # Search for a .env near the source / repo root; never hard-code a key.
    here = Path(__file__).resolve().parent
    for envp in (here / ".env", here.parent / ".env", here.parent.parent / ".env"):
        if envp.exists():
            for ln in envp.read_text().splitlines():
                if "=" in ln and not ln.strip().startswith("#"):
                    k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
            break
    return os.getenv("GEMINI_API_KEY")


def _is_rate_limit(ex):
    s = str(ex).lower()
    return any(t in s for t in ("429", "resource_exhausted", "quota", "rate", "503", "unavailable", "500"))


async def main():
    key = _load_key()
    if not key:
        print("⚠️ sin GEMINI_API_KEY -> se omite"); sys.exit(0)
    try:
        from google import genai
        from google.genai import types
    except Exception as ex:
        print(f"⚠️ google-genai no instalado ({ex})"); sys.exit(0)
    client = genai.Client(api_key=key)
    safety = [types.SafetySetting(category=c, threshold="BLOCK_NONE")
              for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    cfg = types.GenerateContentConfig(response_mime_type="application/json", safety_settings=safety,
                                      temperature=0.7, max_output_tokens=4096)

    g = json.load(open(GEMINI_PRED))
    train = json.load(open(C.TRAIN_JSON, encoding="utf-8"))

    # memes minoritarios (por voto mayoritario de anotadores)
    targets = []  # (meme_id, n_paraphrases, majority_cats)
    for m in train.values():
        mid = str(m["id_EXIST"])
        votes = [v for v in (_ann_cats(x) for x in m["labels_task2_3"]) if v is not None]
        if not votes:
            continue
        n = len(votes)
        maj = [c for c in CATS if sum(1 for v in votes if c in v) / n >= 0.5]
        if "MISOGYNY-NON-SEXUAL-VIOLENCE" in maj:
            targets.append((mid, N_MISOGYNY, maj))
        elif "SEXUAL-VIOLENCE" in maj:
            targets.append((mid, N_SEXUAL, maj))
    print(f"memes minoritarios: {len(targets)}  (esperados ~{sum(t[1] for t in targets)} ejemplos sintéticos)")

    results = json.load(open(OUT)) if OUT.exists() else {}
    pending = [t for t in targets if t[0] not in results and t[0] in g and isinstance(g[t[0]], dict)]
    print(f"{len(results)} en caché, {len(pending)} pendientes")
    if not pending:
        print("✅ nada pendiente"); return

    sem = asyncio.Semaphore(CONCURRENCY)
    save_lock = asyncio.Lock()
    state = {"done": 0, "t0": time.time()}

    async def _save():
        async with save_lock:
            json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)

    async def worker(mid, n, maj):
        d = g[mid]
        prompt = PROMPT.format(n=n, categories=", ".join(maj),
                               description=d.get("description", ""), sexism_analysis=d.get("sexism_analysis", ""),
                               reasoning=d.get("reasoning", ""),
                               category_reasoning=(d.get("task2_3", {}) or {}).get("category_reasoning", ""))
        async with sem:
            await asyncio.sleep(random.uniform(0.02, 0.2))
            delay = 1.0
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.aio.models.generate_content(model=DEFAULT_MODEL, contents=[prompt], config=cfg)
                    out = json.loads(resp.text).get("paraphrases", [])
                    results[mid] = [p for p in out if isinstance(p, dict) and p.get("category_reasoning")]
                    break
                except Exception as ex:
                    if _is_rate_limit(ex) and attempt < MAX_RETRIES - 1:
                        w = delay + random.uniform(0, delay * 0.3)
                        await asyncio.sleep(w); delay *= 2; continue
                    print(f"  [{mid}] error: {ex}"); results[mid] = []; break
            state["done"] += 1
            if state["done"] % 20 == 0:
                el = time.time() - state["t0"]
                print(f"  {state['done']}/{len(pending)}  ({el:.0f}s)", flush=True)
                await _save()

    await asyncio.gather(*(worker(*t) for t in pending))
    await _save()
    tot = sum(len(v) for v in results.values())
    print(f"✅ {len(results)} memes, {tot} paráfrasis -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
