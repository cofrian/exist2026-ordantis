# Guía completa — EXIST 2026 Task 2 (memes): qué hemos hecho y por qué

## 0. Contexto y objetivo

- **Competición:** EXIST 2026, Task 2 (memes). 3 subtareas: 2.1 binaria sexista/no, 2.2 intención (NO/DIRECT/JUDGEMENTAL), 2.3 categorización multi-label (5 categorías).
- **Datos:** 5037 memes (3984 train + 1053 test), bilingüe ES/EN, 6 anotadores por meme, etiquetas suaves (no consenso forzado), señales fisiológicas (ET 24 + HR 4 + EEG 80) de 2–4 sujetos por meme.
- **Métricas oficiales:** ICM (hard) e ICM-soft (soft), no F1. Hasta 3 runs hard + 3 soft por subtarea.
- **Baseline a batir:** paper de Arcos & Rosso (F1=0.722, AUC=0.794 en 2.1; AUC=0.655 en 2.2). Práctica LAB-S7 (Mistral zero-shot solo texto, F1~0.71).
- **Equipo:** Ordantis. Deadline 14 may 2026.

---

## 1. Decisiones estratégicas (transversales a las 3 subtareas)

### 1.1 Soft labels en lugar de voto mayoritario — la decisión clave del proyecto

**El problema.** Cada meme tiene **6 anotaciones humanas** (uno por cada anotador), no una "verdad" única. Por ejemplo, en Task 2.1 un meme puede tener:
- 6 YES / 0 NO → consenso claro de "sexista"
- 4 YES / 2 NO → mayoría sexista, pero hay dudas
- 3 YES / 3 NO → **empate 3-3** (614 memes del train caen aquí — el 15.4%!)
- 2 YES / 4 NO → mayoría no sexista
- 0 YES / 6 NO → consenso claro de "no sexista"

¿Qué hacemos con esas etiquetas para entrenar?

**Cómo lo hace Arcos & Rosso (el paper de referencia).**
Reducen las 6 etiquetas a UNA sola por voto mayoritario, y **descartan los empates 3-3** (porque no hay mayoría). El modelo entrena con un único target binario YES/NO:
```python
# A la Arcos
if count(YES) > count(NO):   target = 1.0  (YES)
elif count(NO) > count(YES): target = 0.0  (NO)
else:                         meme descartado (empate)
```
**Pérdidas de esta aproximación:**
- **Tira el 15.4% del dataset** (614 memes). Y los empates son precisamente los más informativos sobre ambigüedad.
- **Pierde toda la información de incertidumbre**. Un meme con 5 YES / 1 NO se trata igual que uno con 6 YES / 0 NO, aunque uno es claramente más sexista que el otro.
- El modelo aprende a estar **siempre seguro** (todas las salidas tienden a 0.0 o 1.0).

**Cómo lo hacemos nosotros.**
Tratamos las 6 etiquetas como **una distribución de probabilidad** y entrenamos contra ella directamente:
```python
# Soft labels
target_soft = count(YES) / count(no_UNKNOWN)
# Ejemplos:
#   6 YES / 0 NO → target = 1.000  (consenso total YES)
#   5 YES / 1 NO → target = 0.833
#   4 YES / 2 NO → target = 0.667
#   3 YES / 3 NO → target = 0.500  (Arcos descarta esto; nosotros sí entrenamos)
#   2 YES / 4 NO → target = 0.333
#   1 YES / 5 NO → target = 0.167
#   0 YES / 6 NO → target = 0.000  (consenso total NO)
```

Y entrenamos con **`BCE soft`** (Binary Cross-Entropy con target en [0, 1] en vez de {0, 1}):

```python
# La función de pérdida que usamos en todos nuestros modelos
loss = BCEWithLogitsLoss(prob_predicha, target_soft)
#                              ↑              ↑
#                       sigmoid(logit)   float ∈ [0, 1] (proporción real)
```

Matemáticamente la BCE soft es: `L = -[t · log(p) + (1-t) · log(1-p)]` donde `t` ya no es 0 o 1 sino un número entre 0 y 1.

**¿Qué consigue esto?**

1. **No tiramos ningún meme.** Los 614 empates 3-3 ahora son ejemplos válidos con `target = 0.5`. El modelo aprende: "cuando es ambiguo, salida ~0.5". Eso es información, no ruido.

2. **El modelo aprende a expresar incertidumbre.** Si entrenas con BCE soft, los logits del modelo NO tienden a ±∞ (no quiere salidas 0/1 puras). Acaba con `sigmoid(logit) ≈ proporción_real` por construcción. Está **bien calibrado de fábrica**.

3. **ICM-soft se beneficia de esto enormemente.** La métrica oficial soft compara la **distribución** del modelo vs la **distribución real** de los anotadores. Un modelo entrenado con BCE soft acierta esa distribución; uno entrenado con voto mayoritario sale catastrófico en soft (ej. Gemini argmax saca ICM-soft −2.56).

**Ejemplo concreto** del impacto:
- Meme con 5 YES / 1 NO. La verdad soft es {YES: 0.83, NO: 0.17}.
- Modelo A entrenado con BCE soft → predice `{YES: 0.81, NO: 0.19}` ✓ muy cerca → ICM-soft alto.
- Modelo B entrenado con voto mayoritario → predice `{YES: 0.99, NO: 0.01}` ✗ "demasiado seguro" → ICM-soft penaliza.
- Modelo C entrenado con voto mayoritario y descartando empates → no sabe predecir bien cerca del 0.5 → en los memes empatados del test predice 0.99 cuando debería predecir 0.5 → catastrófico.

**Mismo principio para Task 2.2 y 2.3.**
En 2.2 tenemos 3 clases (NO/DIRECT/JUDGEMENTAL). En vez de elegir la mayoritaria, calculamos la **distribución soft de los 6 anotadores**:
```python
# Para un meme con votos [DIRECT, DIRECT, DIRECT, JUDGEMENTAL, NO, NO]:
target_22 = [count(NO)/n, count(DIRECT)/n, count(JUDG)/n] = [0.333, 0.500, 0.167]
```
Y se entrena con BCE soft + Focal Loss contra esa distribución.

En 2.3 (multi-label) cada categoría c tiene su propia distribución marginal:
```python
# Para una categoría c en un meme:
target_c = (nº anotadores que marcaron c) / n_anotadores_válidos
# Cada categoría puede tener un valor en [0, 1] independiente.
```
Y se entrena con BCE soft por categoría + Focal weight.

**Detalle implementación.** En el código:
- `data.soft_label(meme) = count("YES")/len(labels)` (línea 55 de `data.py`).
- En `train._run_epoch`: `tgt = (batch["soft"] >= 0.5).float() if hard_target else batch["soft"]`. Con `hard_target=False` (que es el default y lo que usamos en todos los modelos), `tgt = batch["soft"]` es directamente el float entre 0 y 1.
- El loss: `F.binary_cross_entropy_with_logits(logit, tgt)` — la versión "with logits" es numéricamente estable y acepta cualquier float en [0, 1] como target.

**Modelos del proyecto donde se usa soft labels:**
- M1, M2, las 5 vistas de M3, blend Vista E + Gemini.
- Vista E-2.1 y todas sus variantes (max=512, Longformer, _R).
- Vista E-2.2 y sus variantes.
- Vista E-2.3 y sus variantes.

**El único modelo con target duro:** `M1_baseline_ablation` — entrenado deliberadamente "à la Arcos" (target binario + descartando empates) para que sirva de **fila de comparación** en la tabla de ablación que demuestra cuánto aporta cada mejora.

**Resumen ejecutivo:**
> Mientras Arcos toma las 6 etiquetas y vota una mayoría, nosotros las **tratamos como una distribución de probabilidad**. Esto: (a) gana 614 memes empatados; (b) entrena un modelo bien calibrado; (c) sube ICM-soft (la métrica oficial soft); (d) gestiona honestamente la ambigüedad de los memes — porque la realidad es que **a veces los humanos no se ponen de acuerdo, y eso ES información que el modelo debe aprender**.

### 1.2 Entrenamiento en dos fases
- **Fase 1 (5 ép.):** XLM-RoBERTa congelado, solo se entrena la cabeza (head warm-up con lr=5e-5). Evita que la cabeza arranque siendo ruido cuando XLM-R se descongele.
- **Fase 2 (15 ép.):** XLM-R descongelado, LRs diferenciados (capas bajas 1e-5, altas 3e-5, cabeza 1e-4), early stopping sobre ICM-soft (paciencia 4).
- Es el esquema del paper de Arcos.

### 1.3 Mean-pooling enmascarado en vez de `[CLS]`
- **Origen:** durante el debug inicial, Fase 1 de M1 no superaba AUC 0.57 y un assert estricto abortaba. Tras diagnóstico con prints (las features eran reales, en GPU, sin NaN), la causa real era que **el `[CLS]` de un XLM-RoBERTa sin fine-tunear es una representación de frase pésima** (resultado SBERT, Reimers & Gurevych 2019).
- **Cambio:** sustituido por **mean-pooling enmascarado** de los tokens (suma ponderada por la máscara de padding ÷ longitud válida). Beneficia tanto Fase 1 (warm-up con XLM-R congelado) como Fase 2.
- **Cambio simultáneo:** el sanity-check estricto se movió de Fase 1 (irreal) a Fase 2 ep5 con `assert val_AUC > 0.70`.

### 1.4 Bilingüismo gestionado con un solo modelo
- XLM-RoBERTa-base procesa ES y EN nativamente — no hay modelos separados por idioma.
- Las **emociones Ekman** sí dependen del idioma (modelos `daveni/twitter-xlm-roberta-emotion-es` y `j-hartmann/emotion-english-distilroberta-base`), salida homogénea a 7 dims.
- Split 85/15 estratificado por idioma y por etiqueta soft binarizada.

### 1.5 Gemini 3 Flash como "traductor visual" (no como features)
- **Idea original (Arcos):** captions con Qwen-VL para describir la imagen. Sustituido por **Gemini** (modelo concreto: `gemini-3-flash-preview`, el "Flash" de la familia 3.x).
- **Por qué Flash y no Pro:** primero probamos `gemini-3.1-pro-preview` (el thinking model, máxima calidad), pero tardaba ~10-15 s por meme → ETA ~12 h para los 5037 memes. Cambiamos a `gemini-3-flash-preview`: sin thinking, ~1-2 s por meme, 5037 memes en ~75 min con 15 peticiones concurrentes. La diferencia de calidad en `description` y `sexism_analysis` es marginal para nuestro uso (que solo necesita una descripción correcta de la imagen).
- Una llamada por meme (`precompute_gemini.py`, async con `asyncio.Semaphore(15)`, exponential backoff ante 429/5xx, jitter aleatorio) → JSON con `description`, `sexism_analysis`, `reasoning`, predicciones por subtarea, irony, etc.
- Salida cacheada en `gemini_predictions.json` (5037/5037 válidas, tras 3 reintentos).
- **Cómo se usa:** la `description + sexism_analysis` se concatena al OCR como texto enriquecido para XLM-R. No se usan las probabilidades de Gemini como features dentro del modelo.
- **Coste:** ~$1-3 (Flash es ~10× más barato que Pro).

---

## 2. Pre-cómputos (offline, una vez)

```
┌────────────────────────────────────────────────────────────────┐
│ precompute_emotions.py  → daveni-XLM-R (ES) + j-hartmann (EN)  │
│                           → ekman_emotions.json   (7 dims/meme)│
├────────────────────────────────────────────────────────────────┤
│ precompute.py           → ViT-base congelado (ImageNet)        │
│                           → vit_embeddings.npz   (768/meme)    │
├────────────────────────────────────────────────────────────────┤
│ precompute_gemini.py    → Gemini 3 Flash (API, async, conc=15) │
│                           → gemini_predictions.json (5037/5037)│
└────────────────────────────────────────────────────────────────┘
```

---

## 2.0 Limpieza del texto OCR (preprocesado mínimo)

El campo `text` del JSON ya viene con el texto que la organización extrajo del meme vía OCR — pero **es ruidoso**: trae URLs, hashtags, errores de OCR, emojis y emoticonos, etc. Limpiamos solo lo estrictamente necesario para no quitar señal:

```python
# data.py
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WS_RE  = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = _URL_RE.sub(" ", text)                  # 1) quitar URLs
    t = re.sub(r"#(\w+)", r"\1", t)             # 2) quitar el '#' pero conservar la palabra
    t = _WS_RE.sub(" ", t).strip()              # 3) colapsar espacios múltiples
    return t
```

### Qué SÍ hacemos
| Regla | Por qué |
|---|---|
| **Quitar URLs** (`http://...`, `https://...`, `www.facebook.com/...`) | El texto OCR de memes a menudo incluye una marca de agua o pie con la URL de la web origen (p. ej. `WWW.facebook.com/...`). Es ruido constante que confunde al tokenizer y no aporta nada. |
| **Quitar el `#` de los hashtags, conservar la palabra** | `#MeToo` → `MeToo`. El **contenido** del hashtag sí es informativo (a menudo más que el resto del meme: `#FemiNazi`, `#WomenInScience`, `#KitchenJokes` te dicen explícitamente la postura del autor). Solo eliminamos el carácter `#` porque la mayoría de tokenizers BPE lo separan en su propio sub-token y rompe la palabra que sigue. |
| **Colapsar whitespace múltiple** (`"  \t\n "` → `" "`) | El OCR a veces deja saltos de línea raros entre frases. Limpiar para que el tokenizer no use posiciones en padding interno. |

### Qué NO hacemos (a propósito)
| Cosa | Por qué la dejamos sin tocar |
|---|---|
| **Emojis** (🤔 😂 🚺 ❤️ …) | XLM-RoBERTa los tokeniza directamente (su BPE soporta UTF-8 multi-byte; típicamente cada emoji = 1-3 sub-tokens). Y los emojis **llevan carga semántica fuerte en memes** — el modelo aprende su uso en el fine-tuning. Quitarlos sería tirar señal. |
| **Emoticonos textuales** (`:)`, `:(`, `xD`, `;P` …) | Igual que los emojis: XLM-R los tokeniza, y en memes son indicadores claros del tono (irónico, judgmental, etc.). |
| **Errores de OCR** (`HSY MUCKERES`, `Mlujeress`, palabras pegadas) | El tokenizer BPE es **robusto a errores tipográficos**: una palabra mal escrita simplemente se trocea en más sub-tokens, pero conserva contexto. Intentar corregir el OCR introduciría errores propios. |
| **Mayúsculas / minúsculas** | XLM-R es case-sensitive y aprovecha la mayúscula sostenida (GRITO IRÓNICO) como señal. No lo casefoldeamos. |
| **Acentos** | Igual: XLM-R los entiende nativamente. |
| **Stop-words** | Nunca se quitan en NLP moderno con transformers. |

### Qué pasa con un texto típico
Antes (OCR crudo, meme español):
```
"A VECES QUISIERA IR AL ZUMBA #FitnessLove 💪 pero veo que las que van... 😂😂😂  WWW.fb.com/funnymemes"
```

Después de `clean_text`:
```
"A VECES QUISIERA IR AL ZUMBA FitnessLove 💪 pero veo que las que van... 😂😂😂"
```

Luego eso se concatena con la `description` y el `sexism_analysis` de Gemini, y se pasa al tokenizer de XLM-R con `truncation=True, max_length=320`.

### Por qué no más limpieza
Probamos NO hacerlo (textos OCR brutos): el tokenizer de XLM-R los digiere bien igualmente, los modelos no se rompen. La limpieza actual da una **mejora marginal** (~0.5-1 pt F1) por quitar las URLs repetidas, que sí confundían cuando había batch de memes con el mismo dominio. El resto (hashtag-cleaning, whitespace) es higiene, no impacta apenas.

**Resumen:** el OCR se limpia lo mínimo (URLs + `#` + whitespace). Emojis, emoticonos y errores de OCR se dejan tal cual porque XLM-RoBERTa los procesa bien y son señal.

---

## 2.bis Modelos de emociones (Ekman) — qué son y cómo entran al modelo

Las emociones de Ekman son 7 estados afectivos básicos universales (Paul Ekman, 1970s) que se han convertido en el "lenguaje común" para etiquetar emoción en NLP:

```
anger · disgust · fear · joy · neutral · sadness · surprise
```

Para cada meme calculamos **una distribución de probabilidad sobre esas 7 emociones a partir del texto OCR** (no de la imagen — esa la analiza Gemini). El resultado es un vector `(7,)` con valores en [0, 1] que **suman aproximadamente 1**.

### 2.bis.1 Por qué usamos DOS modelos (uno por idioma)
Los modelos de emociones específicos por idioma sacan resultados más precisos que un modelo multilingüe genérico para esta tarea, así que usamos:

| Idioma | Modelo HuggingFace | Etiquetas que devuelve |
|---|---|---|
| **Español** | `daveni/twitter-xlm-roberta-emotion-es` | anger, disgust, fear, joy, sadness, surprise, **others** |
| **Inglés** | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, **neutral**, sadness, surprise |

Diferencia importante: el modelo ES tiene `others` como séptima clase; el EN tiene `neutral`. Lo unificamos: `others` (ES) se **mapea a `neutral`** en el vector final. Así todos los memes acaban con el mismo orden canónico de 7 dimensiones (`config.EKMAN_ORDER`).

### 2.bis.2 Cómo se calcula (en `precompute_emotions.py`)

```python
def _run_model(model_id, texts):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(C.DEVICE).eval()
    id2label = model.config.id2label

    # Cada modelo tiene su orden interno; lo mapeamos al orden canónico Ekman
    idx_map = {i: EKMAN_ORDER.index("neutral" if id2label[i].lower() == "others" else id2label[i].lower())
               for i in range(len(id2label))}

    results = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(C.DEVICE)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        # Reordenar al orden Ekman canónico
        ordered = np.zeros((probs.shape[0], 7))
        for src, dst in idx_map.items():
            ordered[:, dst] = probs[:, src]
        results.append(ordered)
    return np.concatenate(results)
```

Flujo:
1. Se separan los memes por idioma (`splits["es"]`, `splits["en"]`).
2. Cada bloque se pasa por el modelo correspondiente en batches de 32-64.
3. Las salidas (softmax sobre las clases del modelo) se **reordenan al esquema Ekman común**.
4. Se cachean en `ekman_emotions.json`: `{id_meme: [a, d, f, j, n, s, su]}`.
5. Si el fichero ya existe, no se vuelve a calcular.

### 2.bis.3 Ejemplo real
Meme 110887 ("a veces quisiera ir al zumba pero veo que las que van..."):
```
anger    0.031
disgust  0.037
fear     0.016
joy      0.028
neutral  0.134
sadness  0.332
surprise 0.422   ← sorpresa, prob máxima
suma     ≈ 1.0
```

### 2.bis.4 Cómo entran al modelo (sin transformación adicional)
- **NO se z-scorean** (ya están en [0, 1], normalizadas por softmax).
- Se concatenan **tal cual** al vector de features que entra al MLP final del `MemeClassifier` (o al trunk de Vista E-2.2 / 2.3).
- Aporta 7 dimensiones extra al final del concat:

```
[ text_emb (768) | img_emb (768) | sensores (108 o 256x3) | Ekman (7) ]
                                                              ↑
                                            anger, disgust, fear, joy,
                                            neutral, sadness, surprise
```

### 2.bis.5 ¿Aportan realmente? ¿Mejoras o ruido?
- **Pros:** dan al modelo una pista barata sobre el tono emocional del texto OCR, especialmente útil cuando XLM-R está congelado en Fase 1.
- **Contras / matiz:** una vez XLM-R se fine-tunea (Fase 2), las 7 dimensiones de Ekman acaban siendo prácticamente redundantes con lo que el propio XLM-R aprende del texto. La ablación de M1 (filas `baseline → +A → +A+B → ...`) muestra que su contribución incremental es **pequeña** (~0.5-1 pt de F1).
- **Decisión:** los dejamos en todos los modelos por consistencia y porque cuestan ≈ 0 (precalculados, 7 dims), pero **no son la palanca principal**. La palanca real en 2.1 son las descripciones de Gemini (Vista E), no las emociones.

---

## 2.ter Sensores fisiológicos (ET, HR, EEG) — qué son y cómo entran

La novedad de EXIST 2026 frente a 2025 es que cada meme viene acompañado de **señales fisiológicas** de los 2-4 sujetos humanos que lo miraron en laboratorio mientras se medían sus respuestas. Es una de las pistas del "Human-Centered AI" de la edición.

### 2.ter.1 Las tres modalidades

| Modalidad | Dimensiones | Qué mide | Cuándo ayuda |
|---|---|---|---|
| **ET** (Eye Tracking) | **24 features/sujeto** | Atención visual: cómo recorre el sujeto el meme con la mirada | Memes donde la "carga visual" cambia (irónicos, sutiles) |
| **HR** (Heart Rate) | **4 features/sujeto** | Activación autonómica (excitación, estrés) | Memes emocionalmente intensos |
| **EEG** (Electroencefalografía) | **80 features/sujeto** | Actividad cerebral (engagement, control cognitivo) | Memes con disonancia cognitiva, ironía |

Total: **108 dimensiones por sujeto**, con 2-4 sujetos por meme.

### 2.ter.2 ET en detalle (24 dims)
Estadísticos agregados durante el tiempo que el sujeto vio el meme:

```
3d_eye_states_pupil_diameter_left  [mm]: mean, std, min, max     (4)
3d_eye_states_pupil_diameter_right [mm]: mean, std, min, max     (4)
blinks_duration_ns:                       mean, std, min, max     (4)
blinks_count:                                                       (1)
fixations_duration_ns:                    mean, std, min, max     (4)
fixations_count:                                                    (1)
saccades_duration_ns:                     mean, std, min, max     (4)
saccades_count:                                                     (1)
reaction_time:                                                      (1)
                                                          TOTAL = 24
```

- **Diámetro pupilar:** proxy clásico de carga cognitiva / arousal afectivo (las pupilas se dilatan con la activación).
- **Parpadeos:** disminuyen con la atención sostenida.
- **Fijaciones:** dónde se queda mirando (no tenemos las coordenadas, solo conteo y duraciones).
- **Sacadas:** movimientos rápidos entre fijaciones; más sacadas → más exploración.
- **Reaction time:** cuánto tarda el sujeto en pasar al siguiente meme tras verlo. Tiempo alto → procesamiento más complejo (interpretar la ironía, por ejemplo).

### 2.ter.3 HR en detalle (4 dims)
Datos del Garmin del sujeto durante el estímulo:
```
garmin_hr_mean, garmin_hr_std, garmin_hr_min, garmin_hr_max
```
Captura la **activación autonómica** durante el visionado: contenido emocionalmente intenso o disturbante suele elevar HR y aumentar su variabilidad.

### 2.ter.4 EEG en detalle (80 dims)
**Bandpower** (potencia espectral) en **5 bandas de frecuencia** sobre **16 canales** = 5 × 16 = **80 features**:

| Banda | Hz | Asociada a |
|---|---|---|
| **Delta** | 0.5-4 | sueño profundo, estados muy relajados |
| **Theta** | 4-8 | memoria, atención sostenida, evaluación crítica |
| **Alpha** | 8-13 | relajación, atención reducida (baja con concentración) |
| **Beta** | 13-30 | concentración activa, esfuerzo cognitivo |
| **Gamma** | 30+ | integración multisensorial, atención muy aguda |

- Los 16 canales (`EXG_Channel_0` … `EXG_Channel_15`) son posiciones del scalp del sujeto. Nuestro pipeline los trata como **80 features genéricas** sin mapearlas a posiciones concretas del cerebro (esto fue una de las cosas que NO hicimos, ver "lo rechazado" para la "regla auxiliar EEG").

### 2.ter.5 Normalización: z-score por modalidad
Los rangos de las features son muy distintos entre modalidades (pupila en mm, blinks en nanosegundos, EEG bandpower en valores arbitrarios). Para que el modelo no se confunda:

```python
# En data.py:
def compute_sensor_stats(train_examples, feat_order):
    # Concatenar TODOS los vectores sensoriales del train (todos los sujetos)
    # y calcular μ, σ por feature
    stats = {}
    for mod in ("ET", "HR", "EEG"):
        all_vecs = np.concatenate([e["sensors"][mod] for e in train_examples])
        stats[mod] = {"mean": np.nanmean(all_vecs, axis=0),
                      "std":  np.nanstd(all_vecs,  axis=0).clip(min=1e-6)}
    return stats

def apply_sensor_norm(examples, stats):
    for e in examples:
        for mod in ("ET", "HR", "EEG"):
            e["sensors_z"][mod] = (e["sensors"][mod] - stats[mod]["mean"]) / stats[mod]["std"]
            # NaN (sensor faltante) → 0 (vector ya está en escala z-score)
            np.nan_to_num(e["sensors_z"][mod], copy=False, nan=0.0)
```

Esto garantiza que cada feature de cada modalidad tenga media 0 y desviación 1 sobre el train. Aplicado a train, val y test con los **mismos estadísticos del train** (nunca recalcular en val/test).

### 2.ter.6 Agregación entre sujetos: avg vs Set-Attention Pooling
Cada meme tiene 2-4 sujetos, cada uno con sus 108 dims. Hay dos formas de pasar de "matriz (n_sujetos, 108)" a un vector único para el modelo:

**Opción A — Promedio simple** (M1, M3-Vista D):
```python
sens_avg = sensors_z.mean(axis=0)   # (n_sujetos, 108) → (108,)
```
Sencillo, pero **pierde información**: si un sujeto reacciona mucho y otros no, la señal se diluye.

**Opción B — SetAttentionPool** (M2, M3 vistas A/B/C/E, Vista E-2.2, Vista E-2.3):

```
sujeto 1: (108,) ──► φ (MLP compartida) ──► h_1 (256,)
sujeto 2: (108,) ──► φ                  ──► h_2 (256,)
sujeto 3: (108,) ──► φ                  ──► h_3 (256,)
                                              │
        ┌─────────────────────────────────────┤
        ▼                                     │
   score lineal: s_i = W_a · h_i              │
   softmax con máscara: α_i = softmax(s_i)    │
        │                                     │
        ▼                                     ▼
   vec_modalidad = Σ α_i · h_i  (256,) ◄──────┘
```

- φ es una **MLP compartida** (Linear 108→256 → GELU → Dropout → Linear 256→256 → GELU) que proyecta cada sujeto al mismo espacio de 256 dims.
- Un puntero de atención aprende un score por sujeto (`W_a · h_i`), softmax sobre los sujetos válidos (con máscara que ignora los padding cuando hay menos de 4 sujetos).
- Resultado: **un vector de 256 dims, invariante al número de sujetos** (2, 3 o 4 → mismo tamaño de salida).

Se aplica **una vez por modalidad**: una atención para ET, otra para HR, otra para EEG. Cada modalidad sale como 256 dims → concat 768 dims totales de sensores.

Referencia teórica: Set Transformer (Lee et al., NeurIPS 2019). El paper de Arcos NO usa esto (concatena promedios), nosotros sí.

### 2.ter.7 Cómo se conectan al `MemeClassifier`

```python
# En models.py - MemeClassifier:
for m in ("ET", "HR", "EEG"):
    if self.set_pool:
        vec, alpha = self.pools[m](batch[f"sens_{m}"].float(),     # (B, n_subj, dim)
                                   batch[f"mask_{m}"])             # (B, n_subj) bool
        feats.append(vec)                                          # (B, 256)
    else:
        feats.append(batch[f"sens_{m}_avg"].float())               # (B, dim) promedio
```

Y luego `concat` con texto, imagen y emociones → MLP final.

### 2.ter.8 ¿Realmente ayudan los sensores?
- **Comparación M1 vs M2 (Task 2.1):** M2 = M1 + set-pooling sobre sujetos. M2 saca AUC 0.741 vs M1 0.737, ICM −0.011 vs −0.040. **+0.4 pts AUC, +0.03 ICM**. La diferencia existe pero no es enorme.
- **Vista B (solo EEG) vs Vista D (sin sensores):** AUC 0.72 vs 0.73. **Sin sensores casi igual que con solo EEG.** En este dataset, los sensores aportan poco frente al texto + descripción de Gemini.
- **Vista E (texto + EEG + descripción Gemini) bate a todo:** AUC 0.88. Casi todo viene de la **descripción de Gemini**, no del EEG.
- **Conclusión honesta:** los sensores no son la palanca principal. Ayudan a M2 sobre M1 marginalmente, y los mantenemos en Vista E-2.2 y Vista E-2.3 más por consistencia y por explotar la novedad del dataset que porque den un salto cualitativo. Lo grande es **texto enriquecido con la descripción de Gemini**.

---

## 3. Arquitectura común — `MemeClassifier`

Es el clasificador base que se configura con un dict `cfg = {text, image, et, hr, eeg, caption, set_pool, emotions}`. Todas las "vistas" y modelos de Task 2.1 son configuraciones de este mismo módulo.

```
                       ┌────────────────────────┐
                       │  TEXTO (OCR + opcional │
                       │   caption Gemini)      │
                       └───────────┬────────────┘
                                   ▼
                  ┌──────────────────────────────┐
                  │  XLM-RoBERTa-base (fine-tune)│
                  │     mean-pooling enmascarado │
                  └───────────┬──────────────────┘
                              │ 768
                              ▼
 IMG ─► ViT-base CLS (congelado, precalculado) ───► 768 ───┐
                                                            │
 ET  ─► sensors_z (24, n_sujetos) ──► avg ó SetAttn(256) ──┤
 HR  ─► sensors_z (4, n_sujetos)  ──► avg ó SetAttn(256) ──┤  concat
 EEG ─► sensors_z (80, n_sujetos) ──► avg ó SetAttn(256) ──┤
                                                            │
 Ekman (7 dims, ya en [0,1])  ─────────────────────────────┤
                                                            ▼
                                              ┌─────────────────────┐
                                              │  MLP (concat → 512  │
                                              │   → GELU → drop     │
                                              │   → 1 logit)        │
                                              └──────────┬──────────┘
                                                         ▼
                                                  sigmoid → P(sexista)
```

`SetAttentionPool`: cada sujeto (s_1…s_K) pasa por una MLP compartida φ → h_i; un score lineal + softmax con máscara da pesos α_i; el vector final es Σ α_i·h_i. Invariante al número de sujetos, aprende qué sujeto es más informativo.

---

## 3.5 Variables de entrada al clasificador — referencia detallada

La red neuronal NO ve los memes "en bruto". Las features se pre-procesan y luego se **concatenan en un único vector** que entra a la MLP final. Para que tengas claro qué se mete y de dónde viene, aquí están todas las variables organizadas:

### Tabla maestra de todas las features (todas las subtareas/variantes)

| # | Variable | Dim | Origen | Pre-procesado | En qué modelos se usa |
|---|---|---|---|---|---|
| 1 | **Texto OCR + caption** | (B, L, 768) → 768 | XLM-RoBERTa-base fine-tuneado | mean-pooling enmascarado sobre L tokens | TODOS (es el corazón) |
| 2 | **Imagen ViT** | 768 | ViT-base CLS, congelado, precomputado | embedding directo (ImageNet) | M1, M2, M3 Vistas A/B/C/D (NO en Vista E) |
| 3 | **ET (Eye Tracking)** | 24 / sujeto | Sensores del lab (Tobii) | z-score por feature sobre train; promedio o `SetAttentionPool(256)` entre sujetos | M1, M2, M3 Vista A/C (NO en B/D/E ni en 2.2/2.3) |
| 4 | **HR (Heart Rate)** | 4 / sujeto | Garmin | igual que ET (24→4) | M1, M2, M3 Vista A/C (NO en B/D/E ni en 2.2/2.3) |
| 5 | **EEG** | 80 / sujeto | bandpower 5 bandas × 16 canales | z-score; `SetAttentionPool(256)` o promedio | M1, M2, M3 Vista A/B/E, **Vista E-2.2**, **Vista E-2.3** |
| 6 | **Emociones Ekman** | 7 | Modelos `daveni/twitter-xlm-roberta-emotion-es` (ES) y `j-hartmann/emotion-english-distilroberta-base` (EN) sobre OCR | softmax (ya en [0,1] y suma 1, **sin z-score**) | TODOS |
| 7 | **Gemini-features 2.2** | 7 | Gemini 3-flash zero-shot, campo `task2_2` | extracción directa | **Vista E-2.2** y variantes (no en 2.1 ni 2.3) |
| 8 | **Gemini-features 2.3** | 6 | Gemini 3-flash zero-shot, campos `task2_1.sexist_probability` + `task2_3.category_probabilities` | extracción directa | **Vista E-2.3** y variantes (no en 2.1 ni 2.2) |

### Detalle de cada variable

**1. Texto OCR + caption** — `XLM-RoBERTa-base` fine-tuneado en Fase 2.
- Entrada al tokenizer: `f"{OCR} </s> {descripción Gemini} </s> {análisis Gemini}"` (en Vista E-2.1) o variantes más largas en 2.2/2.3.
- Tokens → 12 capas transformer → vector (L, 768) por meme.
- **Mean-pooling enmascarado** (suma ponderada por `attention_mask` ÷ longitud válida) → vector único 768.
- `max_length` = 256 (Vista E original), 320 (2.2/2.3 original), 512 (variantes max=512), o 1100 (Longformer).

**2. Imagen ViT** — `google/vit-base-patch16-224` congelado.
- Solo se usa en M1/M2/M3 vistas A-D. **NO en Vista E** (porque allí la imagen entra "en texto" vía la descripción de Gemini).
- Embedding del token `[CLS]` del ViT (precomputado offline en `vit_embeddings.npz`).
- Sin fine-tuning.

**3. ET — Eye Tracking** (24 features por sujeto):
- 4 estadísticos × diámetro pupilar izquierdo + 4 derecho = 8
- 4 estadísticos × duración de parpadeos + count = 5
- 4 estadísticos × duración de fijaciones + count = 5
- 4 estadísticos × duración de sacadas + count = 5
- 1 reaction_time
- **Z-scored** sobre el train.
- 2-4 sujetos por meme → `SetAttentionPool` (con atención aprendible) o promedio simple.

**4. HR — Heart Rate** (4 features por sujeto):
- `garmin_hr_mean`, `_std`, `_min`, `_max`.
- Z-scored, mismo agregado entre sujetos.

**5. EEG** (80 features por sujeto):
- **bandpower** en 5 bandas (Delta 0.5-4Hz / Theta 4-8 / Alpha 8-13 / Beta 13-30 / Gamma 30+) × 16 canales = 80 valores.
- Z-scored, `SetAttentionPool(256)` (en todos los modelos de Vista E-2.2/2.3).

**6. Ekman emotions** (7 dims):
- ES → `daveni/twitter-xlm-roberta-emotion-es` (clases: anger, disgust, fear, joy, sadness, surprise, `others`).
- EN → `j-hartmann/emotion-english-distilroberta-base` (clases: anger, disgust, fear, joy, **neutral**, sadness, surprise).
- Unificación: `others` (ES) → `neutral`. Orden canónico: `[anger, disgust, fear, joy, neutral, sadness, surprise]`.
- Output es softmax (suma 1) — **sin z-score adicional**.
- Precomputado en `ekman_emotions.json`.

**7. Gemini-features para Task 2.2** (7 dims):
```python
feat_22 = [
    task2_1.sexist_probability,         # 1 — Gemini cree que es sexista?
    task2_1.confidence,                  # 2 — su confianza
    task2_2.intention_probabilities.NO,   # 3 — prob de NO sexista
    task2_2.intention_probabilities.DIRECT,    # 4 — prob de DIRECT
    task2_2.intention_probabilities.JUDGEMENTAL, # 5 — prob de JUDGEMENTAL
    1.0 if task2_2.irony_detected else 0.0,    # 6 — flag de ironía
    task2_2.irony_confidence,            # 7 — confianza en la ironía
]
```
Cada dim ya en [0, 1]. No se normaliza más. Se concatena al vector final.

**8. Gemini-features para Task 2.3** (6 dims):
```python
feat_23 = [
    task2_1.sexist_probability,              # 1 — sexista sí/no
    task2_3.category_probabilities.IDEOLOGICAL-INEQUALITY,   # 2
    task2_3.category_probabilities.STEREOTYPING-DOMINANCE,   # 3
    task2_3.category_probabilities.OBJECTIFICATION,          # 4
    task2_3.category_probabilities.SEXUAL-VIOLENCE,          # 5
    task2_3.category_probabilities.MISOGYNY-NON-SEXUAL-VIOLENCE,  # 6
]
```

### El vector concatenado por modelo

**Vista E-2.1** (la ganadora de Task 2.1):
```
[text (768) ⊕ EEG_pool (256) ⊕ Ekman (7)] = 1031 dims
                                            ↓ MLP (1031→512→1) → sigmoid → P(YES)
```
**NO usa:** imagen ViT, ET, HR, Gemini features.

**Vista E-2.2** (Task 2.2):
```
[text (768) ⊕ EEG_pool (256) ⊕ Ekman (7) ⊕ Gemini-22 (7)] = 1038 dims
                                                              ↓ HierarchicalHead (binary + type)
                                                              ↓ → P(NO), P(DIR), P(JUDG)
```

**Vista E-2.3** (Task 2.3):
```
[text (768) ⊕ EEG_pool (256) ⊕ Ekman (7) ⊕ Gemini-23 (6)] = 1037 dims
                                                              ↓ trunk(1037→256)
                                                              ↓ head_sex (1) + head_cat (5 sigmoides)
                                                              ↓ → P(sex), P(c|sex) para cada categoría
```

**M1 / M2 (Task 2.1, pipeline original):**
```
[text (768) ⊕ ViT (768) ⊕ ET_avg (24) ⊕ HR_avg (4) ⊕ EEG_avg (80) ⊕ Ekman (7)] = 1651 dims
                                                                                  ↓ MLP → P(YES)
```
(M1 usa promedios; M2 sustituye `*_avg` por `*_pool(256)`.)

### Importancia real de cada variable

A partir de los resultados experimentales (ablaciones implícitas comparando vistas y modelos):

| Variable | Importancia | Evidencia |
|---|---|---|
| **Texto (XLM-R + descripción Gemini)** | 🟢🟢🟢 **CRÍTICA** | Vista E (texto+EEG+Ekman) saca AUC 0.880; sin descripción Gemini (M1/M2 con texto solo OCR) AUC 0.74 → la descripción aporta ~+0.14 |
| **EEG (set-pooling)** | 🟡 marginal | Vista B (solo EEG) AUC 0.72 vs Vista D (sin sensores) AUC 0.73 → diferencia casi 0 |
| **Imagen ViT** | 🟡 marginal | Vista D (sin sensores) ≈ Vista A (todos sensores), el ViT por sí solo no añade tanto cuando ya hay descripción de Gemini |
| **ET + HR** | 🟡 marginal | M2 (con set-pooling sobre los 3) vs M1 (promedio) sube AUC +0.004 — diferencia pequeña |
| **Emociones Ekman** | 🟡 marginal | M1 sin Ekman vs con Ekman: ~+0.5 pts F1. Útil en Fase 1 (warm-up), poco en Fase 2 |
| **Gemini-features (2.2/2.3)** | 🟡 marginal-baja | El contenido ya está capturado vía el texto enriquecido; las dims numéricas aportan un punto extra de información |
| **Mean-pool vs `[CLS]`** | 🟢🟢 IMPORTANTE | El cambio sacó la Fase 1 del bloqueo. Sin esto la red no convergía en Fase 1. |
| **Soft labels (count(YES)/n_anotadores)** | 🟢🟢 IMPORTANTE | Ablación M1 baseline (target duro) vs M1 + soft labels: ICMSoft mejora claramente |

### En una frase
**La señal principal del modelo viene del texto (OCR + descripción de Gemini) procesado por XLM-RoBERTa fine-tuneado. Las demás features (sensores, Ekman, gemini-features numéricas) aportan algo, pero el salto cualitativo de AUC 0.74 → 0.88 entre la pipeline original y Vista E vino de incorporar la descripción de Gemini al texto, no de añadir más modalidades.**

---

## 4. Task 2.1 — los 4 modelos

| Modelo | Texto | Imagen ViT | ET | HR | EEG | set-pool | caption Gemini | seed |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **M1** | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | 42 |
| **M2** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | 42 |
| M3 Vista A | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | 42 |
| M3 Vista B | ✓ | ✓ | – | – | ✓ | ✓ | – | 123 |
| M3 Vista C | ✓ | ✓ | ✓ | ✓ | – | ✓ | – | 2024 |
| M3 Vista D | ✓ | ✓ | – | – | – | – | – | 7 |
| **M3 Vista E** | ✓ | **–** | – | – | ✓ | ✓ | **✓** | 999 |

- **M1**: baseline mejorado (late fusion, sensores promediados).
- **M2**: M1 + atención sobre sujetos (set-pooling).
- **M3**: ensemble de 5 vistas con semillas e inputs distintos; promedio de probabilidades (soft), voto ≥3/5 (hard). Diversidad: A=todo, B=solo EEG, C=conductual, D=sin sensores, E=Gemini-driven.

### El descubrimiento clave: Vista E aplasta
- AUC val: M1 0.737, M2 0.741, vistas A–D ~0.72-0.74, **Vista E 0.880**.
- F1⁺ val: M1/M2 ~0.68, **Vista E 0.79**, **Gemini crudo 0.87**.
- ICM hard: M1/M2/M3-ensemble ~0, **Vista E +0.32**, **Gemini crudo +0.34**.
- ICM-soft: M1/M2 negativos (-0.45 a -0.60), **Vista E +0.46**.

→ Decisión: el ensemble M3 de las 5 vistas **diluía la Vista E con vistas mediocres**, así que en lugar del ensemble usamos Vista E pura y el blend Vista E + Gemini. Por eso los runs finales de 2.1 son: Vista E, blend, Gemini.

---

## 5. Task 2.2 — Vista E adaptada con head jerárquico

### 5.1 Arquitectura

```
   Texto enriquecido:  OCR </s> descripción Gemini </s>
                       INTENTION: intention_reasoning </s>
                       IRONY: ... </s> sexism_analysis </s> reasoning
                                   │
                                   ▼ (max 320 tokens)
                       XLM-RoBERTa (fine-tune, warm-start de Vista E-2.1)
                       mean-pooling                                     ─► 768
   EEG (80, n_sujetos)  → SetAttentionPool(256)                          ─► 256
   Ekman (7)                                                              ─► 7
   Gemini-2.2 features (7): sexist_prob, confidence, NO_prob, DIR_prob,
       JUDG_prob, irony_detected, irony_confidence                       ─► 7
                                   │
                                   ▼ concat (1038 dims)
                       ┌────────────────────────────────┐
                       │     HierarchicalHead           │
                       │                                │
                       │  bin_head  → sigmoid → p_sex   │
                       │  type_head → softmax → p_dir,  │
                       │                       p_judg   │
                       │                                │
                       │  p_NO   = 1 − p_sex            │
                       │  p_DIR  = p_sex · p_dir|sex    │
                       │  p_JUDG = p_sex · p_judg|sex   │
                       └────────────────────────────────┘
```

### 5.2 Loss y entrenamiento
- **Loss combinada:** BCE-with-logits soft para el head binario (target = 1−p_NO_soft) + **Focal Loss** (γ=2.0, peso DIR=1.0/JUDG=2.0) para el head de tipo, enmascarada a los memes con `target_sexist > 0.5`.
- **Weighted Random Sampler** (1/√freq) sobre la mayoritaria → sobre-muestrea JUDGEMENTAL.
- **Warm-start:** XLM-R + EEG pool inicializados desde el checkpoint de **Vista E-2.1** (199 tensores + 6).
- **2 fases:** 3 + 10 épocas, early stopping.

---

## 6. Task 2.3 — Vista E adaptada multi-label

### 6.1 El problema y los datos

Multi-label sobre **5 categorías** de sexismo, jerarquizadas bajo "YES":

```
ROOT
 ├── NO (no sexista)
 └── YES (sexista)
       ├── IDEOLOGICAL-INEQUALITY
       ├── STEREOTYPING-DOMINANCE
       ├── OBJECTIFICATION
       ├── SEXUAL-VIOLENCE
       └── MISOGYNY-NON-SEXUAL-VIOLENCE
```

- Un meme puede pertenecer a **varias categorías a la vez** (multi-label real, no mutuamente excluyentes).
- Si no es sexista → solo "NO".
- En el JSON, `labels_task2_3` es una **lista por anotador**, cada elemento es: (a) una **lista de categorías** que ese anotador marcó, (b) `"-"` si dijo que no era sexista, o (c) `"UNKNOWN"` si no anotó (se descarta).
- En el dataset, las categorías están **muy desbalanceadas**: STEREOTYPING-DOMINANCE y OBJECTIFICATION son las dominantes; SEXUAL-VIOLENCE y MISOGYNY-NON-SEXUAL-VIOLENCE son raras.

### 6.2 Cómo construimos los soft labels
Para cada meme, sobre los `n` anotadores válidos (no UNKNOWN):
- `p_sex = (anotadores que NO dijeron "-") / n` → probabilidad marginal de que sea sexista.
- Para cada categoría `c`: `p_c = (anotadores que incluyeron c en su lista) / n` → probabilidad marginal de la categoría.
- **Observación clave:** la suma `Σ p_c` puede ser > `p_sex` (porque un anotador puede marcar varias categorías), pero cada `p_c ≤ p_sex` siempre.

Estos sirven de target soft (sin tirar ningún meme, ni siquiera los empates).

### 6.3 Arquitectura — `VistaE23`

```
   Texto enriquecido para 2.3:
       OCR </s> descripción Gemini </s>
       CATEGORIES: <lista de categorías que Gemini detectó (o "none")> </s>
       category_reasoning de Gemini </s>
       sexism_analysis de Gemini
                                   │
                                   ▼ (max 320 tokens, truncado al final)
                  XLM-RoBERTa-base (fine-tune)
                  warm-start desde Vista E-2.2 (que a su vez venía de Vista E-2.1)
                  mean-pooling enmascarado                            ─► 768
                                                                       │
   EEG (80, n_sujetos) → SetAttentionPool(256, warm-start de E-2.1)   ─► 256
   Ekman (7)                                                            ─► 7
   Gemini-2.3 features (6):                                             ─► 6
       [ sexist_prob,
         IDEOLOGICAL-INEQUALITY_prob,
         STEREOTYPING-DOMINANCE_prob,
         OBJECTIFICATION_prob,
         SEXUAL-VIOLENCE_prob,
         MISOGYNY-NON-SEXUAL-VIOLENCE_prob ]
                                   │
                                   ▼ concat (≈ 1037 dims)
                        ┌─────────────────────────────────┐
                        │  trunk:                          │
                        │   Linear(1037 → 512) + GELU + dr │
                        │   Linear(512 → 256)  + GELU + dr │
                        └──────────────┬───────────────────┘
                                       │ 256
                          ┌────────────┴─────────────┐
                          ▼                          ▼
              ┌──────────────────┐     ┌────────────────────────────┐
              │  head_sex        │     │  head_cat                  │
              │  Linear(256→1)   │     │  Linear(256→5)             │
              │  → sigmoid       │     │  → 5 sigmoides             │
              │                  │     │  (probabilidad condicional │
              │  p_sex           │     │   p_c|sex para cada cat)   │
              └──────────────────┘     └────────────────────────────┘
                          │                          │
                          ▼                          ▼
              p_NO_marginal = 1 − p_sex
              p_c_marginal  = p_sex · p_c|sex   (para cada categoría c)
```

### 6.4 Loss combinada

```python
# head binario (¿es sexista?)
Ls = BCE_with_logits(logit_sex, p_sex_soft)

# head condicional (categorías | sexista)
target_c|sex = p_c_soft / p_sex_soft   (clamp 0..1)
bce_c = BCE_with_logits(logits_cat, target_c|sex, pos_weight=W_c)
focal_weight = (1 − p_t)^γ            # γ = 2.0
Lc_i = focal_weight · bce_c
Lc   = mean(Lc_i · mask_sexista)       # solo donde el meme es sexista

L = Ls + Lc
```

- `W_c` (pos_weight por categoría) se calcula sobre el train: `((1 − prevalencia) / prevalencia)` con clamp en [0.5, 8.0]. Resultado para los pesos finales:
  - IDEOLOGICAL-INEQUALITY ≈ 1.36
  - STEREOTYPING-DOMINANCE ≈ 1.27
  - OBJECTIFICATION ≈ 1.18
  - **SEXUAL-VIOLENCE ≈ 2.80** ← rara, sube
  - **MISOGYNY-NON-SEXUAL-VIOLENCE ≈ 5.05** ← muy rara, sube fuerte
- El factor focal `(1−p_t)^γ` hace que el modelo se concentre en ejemplos difíciles (los que no acierta) y deje de aprender los fáciles cuando ya los ha pillado.
- Equivale a una **Asymmetric Loss** moderada (penaliza más los falsos negativos en clases raras).

### 6.5 Entrenamiento — cascada de warm-starts

1. **Vista E-2.1** entrena con sus pesos aleatorios sobre Task 2.1 (binaria sexista/no).
2. **Vista E-2.2** **warm-start** desde Vista E-2.1 (carga XLM-R + EEG pool) y entrena sobre Task 2.2.
3. **Vista E-2.3** **warm-start desde Vista E-2.2** (en vez de 2.1) → así el modelo ya viene con XLM-R "entendiendo" la estructura del texto enriquecido (descripción Gemini, categorías, razonamientos) y solo tiene que aprender la salida multi-label.

```
Vista E-2.1  ──warm-start──► Vista E-2.2  ──warm-start──► Vista E-2.3
(sexista sí/no)              (intención NO/DIR/JUDG)       (5 categorías multi-label)
```

- **2 fases:** 3 épocas frozen + 12 épocas full fine-tune (paciencia 4).
- LRs diferenciados (igual que el resto): embeddings + capas bajas 1e-5, capas altas 3e-5, cabeza/trunk 1e-4.
- Métrica para early stopping: F1 micro de las 5 categorías en validación (rápido, robusto, correlaciona con ICM).
- Resultado del entrenamiento: best val F1 micro ≈ 0.575, en ~8-10 épocas.

### 6.6 Calibración + thresholds por categoría con protección a minorías

Tras entrenar, sobre validación:

1. **Platt scaling por categoría:** se ajusta una regresión logística independiente para cada una de las 5 categorías sobre sus logits → suaviza la sobre-confianza del modelo en las clases dominantes.
2. **Thresholds por categoría** optimizados con grid search maximizando `0.5·F1 + 0.5·ICM-Norm`:
   - IDEOLOGICAL-INEQUALITY: `t = 0.21`
   - STEREOTYPING-DOMINANCE:   `t = 0.27`
   - OBJECTIFICATION:           `t = 0.13`
   - **SEXUAL-VIOLENCE:         `t = 0.05`** ← muy bajo, "protección a minoría"
   - **MISOGYNY-NON-SEXUAL-VIOLENCE: `t = 0.13`** ← bajo, protección moderada
- La idea: para las categorías raras, exigir prob > 0.5 nunca dispararía → se baja el umbral para que se prediga aunque la prob sea pequeña pero significativa.

### 6.7 Inferencia y submissions

- **Soft:** `{NO: 1−p_sex, IDEOLOGICAL-INEQUALITY: p_sex·p_c1, ..., MISOGYNY-...: p_sex·p_c5}`. No suma 1 (multi-label). Calibrado con Platt.
- **Hard:** lista de categorías con `p_c_marginal ≥ threshold_c`. Si la lista queda vacía pero el meme se predice sexista → categoría con probabilidad máxima (fallback para que la submission siempre tenga ≥1 valor, como exige el formato).

### 6.8 Resultados en validación

| Sistema | F1 macro | F1 IDEO | F1 STEREO | F1 OBJ | F1 SEX-VIOL | F1 MISO | ICMSoft-proxy |
|---|---|---|---|---|---|---|---|
| Gemini 3-flash crudo | 0.461 | — | — | — | — | — | −0.602 |
| Vista E-2.3 (thr 0.5 uniforme) | 0.575 | — | — | — | — | — | −0.405 |
| **Vista E-2.3 (Platt + thr protegidos)** | **0.581** | **0.673** | 0.628 | 0.600 | **0.482** | 0.520 | **−0.383** |

- **Subida clara sobre Gemini crudo**: +0.12 en F1 macro (0.581 vs 0.461). Esta es la mayor ganancia de "Vista E vs Gemini" en cualquiera de las 3 subtareas (en 2.2 el modelo entrenado apenas movía sobre Gemini; en 2.1 se compensaban).
- **Las minorías levantan**: SEXUAL-VIOLENCE F1 0.48 y MISOGYNY F1 0.52 — sin la protección de umbrales estos bajaban significativamente.
- ICMSoft-proxy −0.38: el mejor que conseguimos para 2.3 — pero **negativo aún**, indica que las probabilidades soft del modelo no llegan al nivel del baseline trivial de EXIST en soft-soft. (En 2.3 multi-label, ICM-soft es muy difícil de hacer positivo.)

### 6.9 Lo que NO hicimos en 2.3 (y por qué)
- **No probamos data augmentation con paráfrasis de Gemini** para las categorías raras (lo que sí intentamos en 2.2 y empeoró). Lección aprendida en 2.2 → no la repetimos en 2.3.
- **No probamos ensemble de seeds.** Vista E-2.3 funcionó a la primera (F1 0.581 vs Gemini 0.461 es una diferencia grande), no había necesidad y el coste era ~6h de cómputo.
- **No usamos features sensoriales ET/HR**, solo EEG. Por consistencia con la arquitectura "tipo Vista E" que demostró ser la ganadora; añadir más sensores habría requerido tunear de nuevo.
- **No exploramos pesos `pos_weight` distintos.** Los calculados con `(1-p)/p` clamp [0.5, 8.0] funcionaron bien al primer intento.

---

## 7. Mejoras de inferencia (post-entrenamiento, sin reentrenar)

### 7.1 Threshold óptimo (Mejora B)
- Para 2.1: barrido del umbral en [0.30, 0.70] maximizando ICM en val.
- Para 2.2: threshold 2D `(t_JUDG, t_DIR)` con regla jerárquica `JUDG > t_JUDG > DIR > t_DIR > NO`. Optimización: F1macro ponderado (JUDG×2) combinado con ICM normalizado.

### 7.2 Temperature scaling (Mejora C, Task 2.1)
- `prob = sigmoid(logit/T)`, ajusta T por mínima cross-entropy soft.
- Resultado interesante: en Vista E-2.1 → **T = 1.000** (ya estaba bien calibrada de fábrica por el entrenamiento con soft labels). Confirmado también con Platt 2-param (a=1, b=0). Por eso lo dejamos sin recalibrar.

### 7.3 Platt scaling por clase (Idea 5, Task 2.2 y 2.3)
- Para multi-clase: cada clase se calibra con una regresión logística independiente sobre los logits → renormalización para que sumen 1.
- Mejoró ICM-soft de Vista E-2.2 sola (−0.679 → −0.439) y del blend (al blend raw ya bueno, el Platt le hizo daño: −0.157 → −0.265; por eso el run 1 soft de 2.2 usa el blend RAW).
- En 2.3 mejoró la calibración por categoría.

### 7.4 TTA (Mejora D, Task 2.1)
- Para los modelos con rama ViT: 5 augmentaciones (estándar + 2 random-crop 232→224 + 2 color-jitter ±0.05, sin flip) → promediar embeddings.
- **No aplica a Vista E** (no usa ViT).

### 7.5 Regla zona dudosa (Mejora E)
- Para predicciones cerca del umbral en M2/M3 de 2.1.

### 7.6 Voto entre vistas (Mejora F, M3 hard)
- Voto mayoritario ≥3/5 (o ≥3/4 sin Vista E).

---

## 8. Lo probado y RECHAZADO — y por qué

| Idea | Qué probamos | Resultado | Por qué la rechazamos |
|---|---|---|---|
| **Captions Qwen-VL** | Captioning con Qwen2.5-VL en lugar de Gemini | No llegó a entrenarse | Sustituido por Gemini (primero `gemini-3.1-pro-preview`, finalmente `gemini-3-flash-preview`). El script existe pero no se usa. |
| **Gemini 3.1 Pro como caption final** | Pipeline arrancado con `gemini-3.1-pro-preview` (modelo "thinking", máxima calidad) | ~10-15 s por meme → ETA ~12 h para los 5037 memes | Demasiado lento para el deadline. Cambiamos a `gemini-3-flash-preview` (sin thinking, ~1-2 s/meme), async con 15 concurrentes + backoff. Resultado: 5037 memes en ~75 min. Calidad de las descripciones suficiente, las predicciones por subtarea siguen siendo precisas. |
| **Ensemble M3 5-vistas como run final 2.1** | Promedio de probabilidades A+B+C+D+E | F1⁺ 0.690, ICM +0.033, ICMSoft −0.251 | El ensemble diluía Vista E (que sola daba F1⁺ 0.79, ICM +0.32). Sustituido en los runs por Vista E pura y blend Vista E + Gemini. |
| **`assert val_AUC > 0.65` tras Fase 1** | Sanity check estricto | M1 abortaba con AUC 0.57 | Expectativa irreal. `[CLS]` de XLM-R congelado da AUC ~0.57 max. Sustituido por mean-pooling + assert movido a Fase 2 ep5 ≥ 0.70. |
| **Data augmentation 2.2 con paráfrasis Gemini** (Idea 2) | 431 memes JUDGEMENTAL → 1276 paráfrasis del OCR vía Gemini async; reentrenar | F1macro 0.439 ↓↓, ICM −0.063 ↓↓ | Las paráfrasis solo cambian el OCR; el resto del texto enriquecido (análisis Gemini) es idéntico → poca diversidad real. **Revertido.** |
| **Focal α=[1, 3.5] γ=2.5** (Idea 3) | Pesos más agresivos a JUDGEMENTAL | F1[JUDG] subía 0.05 pero F1[NO/DIR] caía mucho; ICM hard −0.063 | Sobre-pesa la clase minoritaria a costa de las otras. **Revertido a α=[1, 2.0] γ=2.0.** |
| **Ensemble 5 seeds Vista E-2.2** (Idea 4) | Promediar 5 reentrenamientos | No probado | Coste ~7.5h, mejora esperada modesta. Descartada por tiempo. |
| **Regla auxiliar EEG zona dudosa 2.2** (Idea 7) | Score continuo con `eeg_p8_alpha`, `eeg_fp1_theta`, etc. | No probado | Necesita mapeo concreto de canales EEG (P8 alpha, Fp1 theta) que nuestro pipeline tiene como bandpowers genéricos. Riesgo de empeorar. |
| **Auxiliary head de acuerdo** (Idea 8) | Predecir nº anotadores que coinciden | No probado | Coste 1.5h+reentrenar. Descartada por tiempo, marginal. |

---

## 9. Cambios de criterio importantes (línea temporal)

1. **Inicial:** plan tipo Arcos — M1/M2/M3-ensemble como los 3 runs, Qwen-VL para captions.
2. **Tras debug Fase 1:** `[CLS]` → mean-pooling, assert estricto a Fase 2.
3. **Tras Gemini disponible:** Qwen-VL → primero `gemini-3.1-pro-preview` (con billing activado), pero el modelo "thinking" tardaba ~10-15 s por meme (ETA ~12 h). **Cambiado a `gemini-3-flash-preview`** (async, 15 concurrentes, backoff exponencial, jitter): ~1-2 s/meme, 5037 memes en ~75 min. Vista E ahora come "descripción de Gemini Flash".
4. **Tras evaluar M3 ensemble:** sale **peor que Vista E sola**. Cambio: los runs de 2.1 ya **no son M1/M2/M3-ensemble**, sino **Vista E / blend / Gemini** (los tres mejores en validación).
5. **Tras evaluar Gemini crudo en 2.1:** F1⁺ 0.87, ICM +0.34. Mejor que cualquier modelo entrenado en F1. Lo metimos como un run, sin reentrenar nada.
6. **2.2 — primer intento:** Vista E con head jerárquico + warm-start. F1macro 0.56. Aplicamos Ideas 1 (thr ponderado) + 5 (Platt) → ICM +0.035, ICMSoft −0.44.
7. **2.2 — Ideas 2+3 (intentadas):** aumento de datos + focal agresivo → **empeoró**, revertido. Lección: las mejoras de post-proceso (sin reentrenar) son más rentables aquí.
8. **2.2 — final:** descubrimos que el blend (0.6·VistaE + 0.4·Gemini) **iguala a VistaE en ICM y la supera en F1macro**, y que el blend raw es el mejor soft (−0.157 vs −0.44 de Vista E sola). Los runs finales son blend+thr / VistaE+thr / blend argmax (hard) y blend raw / blend+Platt / VistaE+Platt (soft).
9. **2.3:** modelo nuevo (5 sigmoides multi-label, ASL, warm-start desde 2.2) → bate a Gemini puro con claridad (F1macro 0.581 vs 0.461).

---

## 10. Submission final (zip = 302 KB, 18 ficheros)

| | run 1 (mejor) | run 2 | run 3 |
|---|---|---|---|
| **2.1 hard** | Vista E — ICM **+0.386** · F1⁺ 0.861 | blend (+0.373) | Gemini (+0.343) |
| **2.1 soft** | blend — ICMSoft **+0.596** | Vista E (+0.480) | Gemini (+0.234) |
| **2.2 hard** | blend + thr — ICM **+0.035** · F1macro 0.587 | Vista E + thr (+0.035 · F1 0.560) | blend argmax (−0.045) |
| **2.2 soft** | blend raw — ICMSoft **−0.157** | blend + Platt (−0.265) | Vista E + Platt (−0.439) |
| **2.3 hard** | Vista E (Platt+thr protegido) — F1macro **0.581** | Vista E conservador | Gemini (0.461) |
| **2.3 soft** | Vista E (Platt) — ICMSoft-proxy **−0.383** | Vista E (=) | Gemini (−0.602) |

**Resumen vs referencias:**
- vs paper Arcos (F1=0.722 en 2.1, AUC=0.655 en 2.2): claramente por encima en 2.1 (F1 0.86–0.87). En 2.2 difícil comparar AUC↔F1macro pero el run principal tiene ICM positivo.
- vs práctica LAB-S7 (Mistral zero-shot 2.1, F1~0.71): muy superior (F1 0.86–0.87).

---

## 11. Lecciones aprendidas

1. **`[CLS]` sin fine-tunear es basura.** Mean-pooling enmascarado es trivial y mejor.
2. **Soft labels > voto mayoritario** para ICM-soft, y ganamos los empates 3-3.
3. **Ensemble no siempre mejora.** Promediar un modelo bueno (Vista E) con otros mediocres lo hunde.
4. **Reentrenar con augmentation sintética puede empeorar.** Cuando el cuello de botella es el dataset, las paráfrasis del OCR sin tocar el análisis no añaden señal.
5. **Los tweaks de post-proceso (threshold 2D, Platt) suelen rentar más que tocar el modelo.** Especialmente cuando el modelo ya está bien entrenado con soft labels.
6. **Vista E (XLM-R sobre la descripción de Gemini) es la arquitectura ganadora** para tareas donde "lo que se ve" importa más que cómo se ve. Funciona porque Gemini convierte la imagen en texto rico que un modelo de lenguaje fine-tuneado puede explotar.
7. **Sanity-checks bien puestos** (no irreales) ahorran horas. El assert estricto debe ir donde realmente puede converger el modelo (final de Fase 2), no al principio.
8. **Gemini paid tier es barato y rentable**. Con `gemini-3-flash-preview` el coste por los 5037 memes fue de ~$1-3 (Flash es mucho más barato que Pro). El free tier no llega para 3.x Pro (limit 0). La diferencia de calidad entre Pro thinking y Flash no compensa esperar 12 h.
9. **La métrica oficial (ICM) penaliza la sobre-confianza.** Modelos entrenados con BCE soft están bien calibrados; modelos zero-shot (Gemini argmax) no, por eso su soft es catastrófico (ICMSoft −2.56).
10. **Una sola submission por equipo** — verificar todo antes de subir.

---

## 12. Diagnóstico del truncamiento de tokens (mayo 13)

Tras la entrega inicial, medimos cuántos memes superaban el `max_length` configurado en cada subtarea (XLM-RoBERTa-base tiene un límite duro de 512 tokens):

| Subtarea | `max_length` actual | mediana | p95 | máximo | % truncados |
|---|---|---|---|---|---|
| 2.1 (Vista E) | 256 | 210 | 311 | 781 | 18.6% |
| **2.2 (Vista E-2.2)** | 320 | 378 | 510 | 977 | **82.1% ⚠️** |
| 2.3 (Vista E-2.3) | 320 | 299 | 434 | 1041 | 37.7% |

**Descubrimiento crítico:** la 2.2 estaba truncando el 82% de los memes, perdiendo una mediana de 73 tokens — generalmente del final del texto enriquecido (`reasoning`, `sexism_analysis`). El modelo veía texto cortado en 4 de cada 5 ejemplos.

Esto motivó la nueva ronda de experimentos: probar `max_length` más largo (hasta 512 con XLM-R-base, y hasta 1100 con un modelo Longformer multilingüe).

---

## 13. Experimentos finales (Vista E ampliada)

Reentrenamos los modelos de las 3 subtareas con dos variantes de arquitectura, **sin tocar la pipeline original** (todo en `_alt/`):

### 13.1 Variante "max=512"
- Sigue siendo **XLM-RoBERTa-base** (mismo backbone que la pipeline original).
- `max_length = 512` (el máximo nativo).
- Para 2.2 inicialmente quitamos el `reasoning` del texto enriquecido por entender que era redundante (luego se descartó).
- **Warm-start** desde Vista E-2.1 (compatible, misma arquitectura).
- Batch reducido a 8 (de 16) por la longitud de secuencia.

### 13.2 Variante "Longformer-4096"
- Backbone nuevo: **`markussagen/xlm-roberta-longformer-base-4096`** (atención esparsa estilo Longformer, multilingüe, 4096 tokens de contexto).
- `max_length = 1100` (cubre el máximo real observado en los datos, 1041 tokens).
- Sin warm-start (arquitectura distinta, pesos no compatibles).
- Batch 4 por el coste mayor por secuencia.

### 13.3 Variante "_R" (con `reasoning` re-añadido)
Para diferenciar el efecto de `max_length` del efecto del `reasoning`, también entrenamos versiones con el campo `reasoning` de Gemini explícitamente incluido (`_R`). Así pudimos comparar: ¿cuánto importa el contexto largo? ¿cuánto importa el contenido del reasoning?

### 13.4 Fix del Weighted Random Sampler en 2.3
Inicialmente los scripts experimentales de 2.3 no incluyeron el `WeightedRandomSampler` que sí usaba la pipeline original — comparación injusta. Las versiones `_v2` lo añaden: sobre-muestrean memes donde aparecen las categorías minoritarias (SEXUAL-VIOLENCE, MISOGYNY-NSV).

### 13.5 Resultados consolidados (validación, 598 memes)

**Task 2.1 (binario)**
| Variante | ICM | F1+ | AUC | ICMSoft |
|---|---|---|---|---|
| Original Vista E (max=256) | +0.386 | 0.861 | 0.880 | +0.480 |
| blend Vista E + Gemini | +0.373 | 0.856 | 0.889 | +0.596 |
| max=512 | +0.394 | 0.872 | 0.885 | +0.302 |
| **max=512 + reasoning (R)** | **+0.411** ⭐ | **0.882** | **0.893** | +0.387 |
| Longformer | +0.401 | 0.868 | 0.883 | **+0.542** |
| Longformer + R | +0.380 | 0.879 | 0.884 | +0.533 |

→ El mejor en hard es **max=512+R**. En soft sigue ganando el **blend Vista E + Gemini**.

**Task 2.2 (intención)**
| Variante | ICM | F1macro | F1 JUDG | ICMSoft Platt |
|---|---|---|---|---|
| Original (max=320) + thr | +0.035 | 0.560 | 0.260 | −0.439 |
| max=512 (sin reasoning) | +0.075 | 0.566 | 0.259 | −0.438 |
| **Longformer (con reasoning)** | **+0.079** ⭐ | **0.602** | **0.374** ⭐ | **−0.329** ⭐ |
| max=512 + R | +0.008 | 0.569 | 0.340 | −0.439 |

→ El **Longformer es claramente el mejor en todo**, especialmente en F1 JUDGEMENTAL (+50% respecto al original). El truncamiento masivo del 82% en la pipeline original era el cuello de botella, no la arquitectura.

**Task 2.3 (categorías multi-label)** — con fix del WeightedSampler
| Variante | F1micro | F1macro | ICM hard | ICMSoftNorm |
|---|---|---|---|---|
| Original (max=320, sampler) | 0.550 | 0.581 | — | proxy ≈ 0.43 |
| max=512 (sin sampler) | 0.624 | 0.714 | +0.339 | 0.209 |
| **Longformer (sin sampler)** | 0.589 | 0.705 | +0.265 | **0.279** ⭐ |
| **max=512 + sampler (v2)** | **0.626** ⭐ | **0.715** ⭐ | **+0.340** ⭐ | 0.121 |
| Longformer + sampler (v2) | 0.619 | 0.702 | +0.240 | 0.176 |
| max=512 + sampler + R | 0.616 | 0.714 | +0.333 | 0.111 |
| Longformer + sampler + R | 0.616 | 0.711 | +0.300 | 0.154 |

→ **F1macro sube de 0.581 → 0.715** con `max=512 + sampler`. Es el salto más grande de los 3 experimentos (+0.13 absoluto, +23% relativo). El truncamiento del 38% y el `max_length` corto eran limitantes reales.

→ Trade-off claro: **el sampler ayuda en hard** (boost minorías → mejor F1macro/ICM) pero **empeora soft** (modelo más confiado → ICMSoft más penalizado). Por eso para soft usamos la variante **sin sampler**.

→ El **reasoning no aporta** en 2.3 (todas las `_R` salen iguales o peores). Lección: el contenido extra del reasoning no añade señal cuando ya se incluye `category_reasoning` específico de Gemini.

---

## 14. Bug encontrado en PyEvALL (sigma=0)

Al re-evaluar 2.3 con la métrica oficial ICMSoft, PyEvALL crashaba con:
```
StatisticsError: cdf() not defined when sigma is zero
```

**Causa:** PyEvALL llama a `NormalDist(mu, sigma).cdf(x)` donde `mu` y `sigma` se computan sobre los valores soft del gold. Si alguna categoría rara (p. ej. `SEXUAL-VIOLENCE`) tiene **gold = 0 en todos los memes de validación**, entonces `sigma = 0` → crash.

**Fix:** monkeypatch local que clampa `sigma` a `max(σ, 1e-9)` antes de invocar `NormalDist`:

```python
from pyevall.metrics.metrics import ICMSoft
def _safe_get_prob_class(self, tupla, comparator):
    if tupla is None or not tupla[0]: return 0
    if tupla[0] not in self.gold_average: return -math.log2(1/len(comparator.gold_df))
    if tupla[1] == 0.0: return 0.0
    sigma = max(float(self.gold_deviation[tupla[0]]), 1e-9)  # ← FIX
    try:
        prob = 1 - NormalDist(mu=self.gold_average[tupla[0]], sigma=sigma).cdf(tupla[1])
    except: return -math.log2(1/len(comparator.gold_df))
    if prob <= 0: return -math.log2(1/len(comparator.gold_df))
    return -math.log2(prob)
ICMSoft.get_prob_class = _safe_get_prob_class
```

Sin tocar el paquete instalado. Aplicable también si la organización usara la misma versión de PyEvALL en su evaluación oficial (no nos afecta a nosotros porque el bug salta solo cuando `sigma=0`, condición que solo se da con categorías rarísimas; el comportamiento "normal" no cambia).

---

## 15. Iteración de zips: v1 → v2 → v3 → v4

A medida que evaluábamos las variantes, fuimos generando submission zips diferentes según el criterio:

| Zip | Tamaño | Filosofía | Cuándo |
|---|---|---|---|
| `exist2026_Ordantis.zip` | 299 KB | Versión inicial (Vista E + blend + Gemini) | tras pipeline original |
| `exist2026_Ordantis_v2.zip` | 215 KB | "El mejor por métrica pura" (top-3 por ICM/ICMSoft estricto) | tras experimentos max=512 y Longformer |
| `exist2026_Ordantis_v3.zip` | 208 KB | "Máxima diversidad de modelos" (3 paradigmas distintos por run) | para robustez ante fallos en test |
| **`exist2026_Ordantis_v4.zip`** ⭐ | **261 KB** | **"Mejores por ICM, swap solo si AUC/F1 muy superior"** (la regla final del usuario) | la entrega final |

Solo se subirá uno (regla de la competición), pero los demás quedan como referencias.

---

## 16. Selección final del v4 — composición por (subtarea, tipo)

### TASK 2.1 — Identificación binaria
| Run | Modelo | ICM | F1+ | AUC | ICMSoft |
|---|---|---|---|---|---|
| hard_1 | **Vista E max=512 + reasoning** | **+0.411** | **0.882** | **0.893** | — |
| hard_2 | Vista E Longformer | +0.401 | 0.868 | 0.883 | — |
| hard_3 | Vista E max=512 (sin reasoning) | +0.394 | 0.872 | 0.885 | — |
| soft_1 | **blend Vista E + Gemini** | — | — | 0.889 | **+0.596** |
| soft_2 | Vista E Longformer | — | — | 0.883 | +0.542 |
| soft_3 | Vista E Longformer + reasoning | — | — | 0.884 | +0.533 |

→ Tres variantes de Vista E entrenadas (max=512, max=512+R, Longformer) cubriendo distintos balances entre ICM y AUC.

### TASK 2.2 — Intención
| Run | Modelo | ICM | F1macro | F1 JUDG | ICMSoft |
|---|---|---|---|---|---|
| hard_1 | **Vista E-2.2 Longformer + thr** | **+0.079** | **0.602** | **0.374** | — |
| hard_2 | Vista E-2.2 max=512 + thr | +0.075 | 0.566 | 0.259 | — |
| hard_3 | blend Vista E-2.2 + Gemini + thr | +0.035 | 0.587 | 0.340 | — |
| soft_1 | **blend Vista E-2.2 + Gemini RAW** | — | 0.587 | 0.340 | **−0.157** |
| soft_2 | Vista E-2.2 Longformer + Platt | — | 0.602 | 0.374 | −0.329 |
| soft_3 | Vista E-2.2 max=512 + Platt | — | 0.566 | 0.259 | −0.438 |

→ El **Longformer es el ganador claro en hard** gracias a procesar el texto sin truncar (82% del original sufría truncamiento). En soft, el **blend raw** es el mejor porque está mejor calibrado (el ensemble con Gemini suaviza la sobre-confianza).

### TASK 2.3 — Categorización multi-label
| Run | Modelo | ICM hard | F1macro | ICMSoftNorm |
|---|---|---|---|---|
| hard_1 | **Vista E-2.3 max=512 + Sampler (v2)** | **+0.340** | **0.715** | — |
| hard_2 | Vista E-2.3 max=512 sin Sampler | +0.339 | 0.714 | — |
| hard_3 | Vista E-2.3 max=512 + Sampler + R | +0.333 | 0.714 | — |
| soft_1 | **Vista E-2.3 Longformer sin Sampler** | — | — | **0.279** |
| soft_2 | Vista E-2.3 max=512 sin Sampler | — | — | 0.209 |
| soft_3 | Vista E-2.3 Longformer + Sampler (v2) | — | — | 0.176 |

→ Los runs hard usan los 3 mejores por ICM (los 3 son XLM-R-base max=512 — el F1/ICM del mejor Longformer queda lejos, no entra por la regla "swap solo si muy superior"). En soft, las 3 mejores combinaciones son **sin sampler** (mejor calibración) — la única regla que cambiamos entre hard y soft.

---

## 17. La regla de selección del usuario: "best ICM unless AUC/F1 very superior"

Para asignar los modelos a los runs aplicamos esta regla:

1. **Por defecto: top-3 estricto por la métrica oficial** (ICM para hard, ICMSoft para soft).
2. **Excepción:** si un modelo no entra en el top-3 por ICM pero tiene un F1 macro o AUC **muy superior** al peor de los top-3, se sustituye al de peor ICM.
3. "Muy superior" se interpretó como diferencia ≥ +0.05 absoluto en F1macro o AUC — diferencias menores no se consideran.

Resultado: la regla solo afectó a la decisión de **no swappear** en Task 2.3 hard (el mejor Longformer estaba a −0.033 ICM del peor max=512 pero solo a −0.003 F1macro — no es "muy superior"). En las demás, los top-3 ICM ya incluían los mejores F1/AUC.

---

## 18.cero CAMBIOS Y DIFERENCIAS RESPECTO AL PAPER DE ARCOS & ROSSO

Este resumen consolidado lista **todo lo que hicimos distinto** al paper de referencia, organizado por categoría. Es la sección que responde "¿qué hemos aportado de nuestra propia cosecha?". Las marcas:
- 🟢 = mejora significativa de métrica
- 🟡 = mejora marginal o mejora-según-contexto
- 🔵 = mismo principio pero implementación distinta
- ⚙️ = detalle de implementación

### A. Gestión de las etiquetas

| # | Arcos | Nuestro proyecto | Impacto |
|---|---|---|---|
| 1 | Voto mayoritario sobre los 6 anotadores → 1 etiqueta dura YES/NO | **Soft labels** = `count(YES)/n_anotadores` ∈ [0, 1] | 🟢 sube ICMSoft fuerte |
| 2 | **Descarta los 614 memes con empate 3-3** (15.4% del dataset) | **Conservamos todos** los memes, con target = 0.5 para empates | 🟢 +15% del dataset disponible |
| 3 | BCE binaria | **BCE soft** (`binary_cross_entropy_with_logits` con target float) | 🟢 modelo bien calibrado de fábrica |
| 4 | Solo evaluación hard | Soft + hard, con calibración separada para soft | 🟢 cubrimos las dos métricas oficiales |

### B. Arquitectura — el cambio más grande: Vista E

| # | Arcos | Nuestro proyecto | Impacto |
|---|---|---|---|
| 5 | **Imagen → ViT-base (CLS token)** como vector visual | **Imagen → Gemini 3-flash** → texto descriptivo → XLM-R | 🟢🟢 AUC 0.74 → 0.88 (+0.14) |
| 6 | Captions con **Qwen2.5-VL** | Captions con **Gemini 3-flash** vía API async | 🟢 calidad mejor, sin entrenar modelo grande |
| 7 | Representación de frase: token **`[CLS]`** | **Mean-pooling enmascarado** (SBERT) | 🟢🟢 desbloquea Fase 1 (resultado SBERT, 2019) |
| 8 | **Sensores concatenados** (promedio entre sujetos o concat plano) | **SetAttentionPool** entre sujetos con atención aprendible | 🟡 +0.004 AUC (M2 vs M1); invariante al nº de sujetos |
| 9 | Head plano: para 2.2, softmax 3 clases directamente | **HierarchicalHead**: `bin_head` (¿sexista?) + `type_head` (DIRECT/JUDG condicional) | 🟢 captura mejor la jerarquía YES → {DIR, JUDG} |
| 10 | Head plano: para 2.3, 5 sigmoides independientes | **head_sex + head_cat conditional**: `P(c) = P(sex) · P(c\|sex)` | 🟢 respeta que la cat. solo existe si es sexista |

### C. Entrenamiento

| # | Arcos | Nuestro proyecto | Impacto |
|---|---|---|---|
| 11 | 2 fases (head warm-up + full fine-tune) con LRs diferenciados | **Igual** — 5 ép. fase 1 + 15 ép. fase 2, LRs (1e-5/3e-5/1e-4) | 🔵 mismo esquema |
| 12 | Sampler uniforme | **Weighted Random Sampler** para boost minorías (JUDGEMENTAL en 2.2, SEX-VIOL y MISO-NSV en 2.3) | 🟢 +F1 minoritarias en hard (pero ⚠️ empeora soft) |
| 13 | Pérdida estándar | **Focal Loss** (γ=2.0) con `pos_weight` por clase ([1.0, 2.0] para DIR/JUDG) | 🟢 +F1 minoritarias |
| 14 | Sin pos_weight diferenciado | **`pos_weight` por categoría** en 2.3 (SEX-VIOL 2.80, MISO-NSV 5.05) | 🟢 +recall en categorías raras |
| 15 | Pérdida = BCE estándar | **BCE + Focal en 2.3** ≈ Asymmetric Loss moderada | 🟢 mejor en multi-label desbalanceado |
| 16 | Texto OCR a secas | **Texto enriquecido**: `OCR </s> descripción Gemini </s> análisis </s> reasoning` | 🟢 Vista E saca AUC 0.88 |
| 17 | `max_length=128` (o 256) | **`max_length=256/512/1100`** según subtarea; 512 para 2.2 y 2.3, Longformer-4096 para variante 2.2 | 🟢 elimina 82% de truncamiento en 2.2 |
| 18 | XLM-R-base | **XLM-R-base + variante con Longformer-4096** (`markussagen/xlm-roberta-longformer-base-4096`) en 2.2 | 🟢 F1macro 2.2 sube 0.560 → 0.602 |
| 19 | Sin warm-start entre tareas | **Warm-start en cascada**: Vista E-2.1 → Vista E-2.2 → Vista E-2.3 | 🟢 convergencia más rápida en 2.2 y 2.3 |
| 20 | Sin gradient checkpointing | **Gradient checkpointing activado** en XLM-R fine-tune | ⚙️ permite batch 16 con bf16 sin OOM |
| 21 | Sin sanity check explícito | **`assert val_AUC > 0.70` tras Fase 2 ep5** (M1 strict_phase2) | ⚙️ aborta temprano si el modelo no aprende |

### D. Inferencia y post-procesado

| # | Arcos | Nuestro proyecto | Impacto |
|---|---|---|---|
| 22 | Threshold 0.5 fijo | **Threshold óptimo barrido en val** [0.30, 0.70] maximizando ICM | 🟢 +0.02-0.04 ICM hard |
| 23 | Sin temperature scaling | **Platt scaling por clase** post-entrenamiento para soft | 🟢 mejora ICMSoft |
| 24 | Sin TTA | **TTA: 5 augmentaciones** en ViT (estándar + 2 random-crop + 2 color-jitter) en M1/M2/M3 | 🟡 marginal en 2.1 (no aplica a Vista E) |
| 25 | Sin regla zona dudosa | **Mejora E: regla zona dudosa** para predicciones cerca del umbral | 🟡 marginal |
| 26 | Argmax simple para 2.2 | **Threshold 2D ponderado** `(t_JUDG, t_DIR)` con regla jerárquica + factor 2× JUDG | 🟢 +F1 JUDG |
| 27 | Threshold único | **Thresholds por categoría con protección a minorías** en 2.3 (SEX-VIOL t=0.05, MISO-NSV t=0.13) | 🟢 recall minorías sube notablemente |
| 28 | Submission de un solo modelo | **Blend Vista E + Gemini** (ensemble 0.5/0.5 de modelo entrenado + zero-shot) en runs | 🟢 mejor ICMSoft +0.596 en 2.1 |
| 29 | No usa zero-shot LLM como baseline | **Gemini crudo** como run propio (uno de los 3) — paradigma diferente, robustez | 🟡 paradigma diverso |

### E. Pre-cómputo y datos

| # | Arcos | Nuestro proyecto | Impacto |
|---|---|---|---|
| 30 | Texto OCR sin limpiar | **Limpieza ligera**: quita URLs (`http://…`, `www.`) + `#` (conserva la palabra) + colapsa whitespace. **Conserva emojis y emoticonos.** | ⚙️ XLM-R los digiere bien |
| 31 | Sin emociones | **Ekman 7-dim** del texto OCR con modelos específicos por idioma (`daveni/...` ES, `j-hartmann/...` EN) | 🟡 marginal en Fase 2 |
| 32 | Caption por meme con un LLM en lotes | **Async con `asyncio.Semaphore(15)`** + exponential backoff + jitter + retry de fallidos | ⚙️ 5037 memes en ~75 min con `gemini-3-flash-preview` |
| 33 | Sensores ya agregados por el dataset | **Z-score por modalidad** sobre stats del train; aplicación a val y test | ⚙️ asegura escala homogénea |
| 34 | Split aleatorio | **Split 85/15 estratificado por idioma Y por etiqueta soft binarizada** | ⚙️ mantiene balance ES/EN y positivos/negativos |
| 35 | Posible modelo por idioma | **Un único modelo bilingüe** (XLM-R multilingüe) — solo Ekman elige internamente | ⚙️ más simple, sin perder calidad |

### F. Estructura del proyecto

| # | Arcos | Nuestro proyecto | Impacto |
|---|---|---|---|
| 36 | M1 + M2 + ensemble M3 con voto mayoritario | Empezamos así, **descartamos M3 ensemble** en favor de **Vista E pura** porque diluía la Vista E ganadora | 🟢 +0.14 AUC |
| 37 | Run hard + soft = modelo binario para hard, dist. soft para soft | **Mismo modelo subyacente para hard y soft** — solo cambia la regla de decisión final | 🔵 más coherente |
| 38 | (No mencionado) | **Sanity checks** durante entrenamiento (grad norm, NaN, val loss vs train loss) + comprobación blanda Fase 1 | ⚙️ debug más rápido |
| 39 | Submission = predicciones del mejor modelo | **3 runs distintos** por cada (subtarea × tipo) con criterio "top-3 por ICM + diversidad" | 🔵 más robusto a fallos de un modelo en test |

### G. Errores nuestros que detectamos y arreglamos (no estaban en Arcos)

| # | Bug encontrado | Causa | Fix |
|---|---|---|---|
| 40 | M1 abortaba en Fase 1 (`AUC=0.57 < 0.65`) | `[CLS]` sin fine-tunear no separa clases | Cambio a mean-pooling + assert movido a Fase 2 ep5 |
| 41 | Augmentation con paráfrasis empeoraba (probado en 2.2) | Las paráfrasis solo cambian el OCR, no el análisis de Gemini → poca diversidad real | Revertido, no usado en la entrega |
| 42 | Focal α=[1, 3.5] γ=2.5 en 2.2 hundía F1 NO/DIR | Sobre-pesa JUDG | Vuelta a α=[1, 2.0] γ=2.0 |
| 43 | `pgrep -f "task22_max512"` se auto-matcheaba en chains | el script de chain contiene esa cadena como texto | Cambio a chains directas |
| 44 | PyEvALL crashaba con `sigma=0` para categorías raras en 2.3 | `NormalDist(0)` no soporta σ=0 | Monkeypatch local con `σ = max(σ, 1e-9)` |
| 45 | task22_longformer.log soft_1 sin Platt scaling (descuido) | Ennl primer build del v2 olvidé aplicar Platt al longformer soft | Corregido en v3/v4 |
| 46 | Original `task23.py` evaluaba ICMSoft con un "proxy" no oficial | Bug PyEvALL impedía la métrica real | Reevaluado tras el fix |

---

## 18.bis EL MODELO, PASO A PASO — qué entra a la red en cada etapa

Esta sección es **el resumen central**: cómo es la red, qué entra, y cómo evolucionó desde la primera versión hasta la que entregamos.

### A. La red, vista a alto nivel

La red neuronal NO es un único bloque. Es una **función** que recibe varios bloques de datos pre-procesados y los combina en una probabilidad. En pseudocódigo:

```
P(meme es sexista)  =  red_neuronal(
                          texto_del_meme,           ← OCR + (opcionalmente) descripción de Gemini
                          imagen_del_meme,          ← embedding ViT precomputado (opcional)
                          señales_fisiológicas,     ← ET / HR / EEG de 2-4 sujetos que lo vieron
                          emociones,                ← softmax 7-dim del modelo de emociones
                          features_de_Gemini        ← solo en 2.2 y 2.3
                       )
```

Por dentro la red tiene **dos partes**:
1. **Codificadores** — convierten cada entrada heterogénea (texto, imagen, sensores...) en un vector de dimensión fija.
2. **Cabeza (head)** — concatena todos esos vectores y los pasa por una MLP que escupe la(s) probabilidad(es) final(es).

Diagrama:

```
                                                    Codificadores
                                                    (cada uno → vector fijo)
                          ┌─────────────┐
   OCR + caption  ──────► │  XLM-R-base │ ─── mean-pool ─►  768
                          │ (fine-tune) │
                          └─────────────┘                    │
                                                              ▼
   Imagen          ──────► ViT-base (congelado)      ─────►  768
                                                              │
                                                              ▼
   ET (n_sub,24)   ──────► SetAttentionPool / mean   ─────►   256 ó 24
                                                              │
                                                              ▼
   HR (n_sub,4)    ──────► SetAttentionPool / mean   ─────►   256 ó 4
                                                              │
                                                              ▼
   EEG (n_sub,80)  ──────► SetAttentionPool / mean   ─────►   256 ó 80
                                                              │
                                                              ▼
   Emociones (7)   ──────► identidad                 ─────►   7
                                                              │
                                                              ▼
   Gemini-feat     ──────► identidad                 ─────►   6 ó 7
                                                              │
                                                              ▼
                            ┌──────────────────────────────────┐
                            │ CONCATENAR → vector grande       │
                            └──────────────────────────────────┘
                                              │
                                              ▼
                                         CABEZA (MLP)
                                              │
                                              ▼
                                       sigmoid / softmax
                                              │
                                              ▼
                                  P(YES) / P(NO,DIR,JUDG) / etc.
```

### B. Las 4 etapas del proyecto

**ETAPA 1 — Punto de partida: M1 (multi-modal late fusion)**

Inspirado en el paper de Arcos. Recibía absolutamente todo lo disponible:

```
M1 ← texto OCR (sin Gemini todavía)
    + imagen ViT
    + ET promediado entre sujetos
    + HR promediado entre sujetos
    + EEG promediado entre sujetos
    + emociones Ekman

Total: 768 + 768 + 24 + 4 + 80 + 7 = 1651 dims → MLP(1651→512→1) → sigmoid → P(YES)
```

**Resultado**: AUC 0.737, F1+ 0.669, ICM −0.040. Por debajo del paper (que reportaba AUC 0.794 / F1 0.722) y por debajo de tu práctica con Mistral zero-shot (~0.71). **Mediocre.**

---

**ETAPA 2 — M2: añadir set-pooling con atención sobre sujetos**

Misma entrada que M1, pero los sensores ya **no se promedian**. Cada sujeto pasa por una MLP compartida y una capa de atención decide qué sujeto pesa más. Saca un vector fijo de 256 dims por modalidad sensorial.

```
M2 ← lo mismo que M1, pero:
       ET (n_sub, 24)  → SetAttentionPool → 256
       HR (n_sub, 4)   → SetAttentionPool → 256
       EEG (n_sub, 80) → SetAttentionPool → 256

Total: 768 + 768 + 256 + 256 + 256 + 7 = 2311 dims
```

**Resultado**: AUC 0.741 (+0.004), ICM −0.011. **Mejora marginal.** El set-pooling ayuda un poco pero no es la palanca principal.

---

**ETAPA 3 — Vista E: el salto cualitativo (Gemini convierte la imagen en texto)**

Mismo concepto de red, pero la entrada cambia DRÁSTICAMENTE: **la imagen ya no entra como embedding de ViT, sino como texto** generado por Gemini ("Description: A meme of Miss Piggy looking backwards... Sexism Analysis: The meme objectifies...").

```
Vista E-2.1 ← texto enriquecido:
              "OCR del meme </s> Description: <Gemini> </s> Sexism Analysis: <Gemini>"
              → XLM-R fine-tuneado → mean-pool → 768

            + EEG con SetAttentionPool → 256
            + emociones Ekman → 7

            (NO usa imagen ViT — Gemini ya describe la imagen en texto)
            (NO usa ET ni HR — solo EEG)

Total: 768 + 256 + 7 = 1031 dims → MLP → sigmoid → P(YES)
```

**Resultado**: AUC **0.880** (+0.14 sobre M2), F1+ 0.792, ICM **+0.323**, ICMSoft **+0.457**. **APLASTANTE.**

**¿Por qué funciona tanto?**
- Las imágenes de memes contienen texto + escena que es muy difícil para un encoder visual genérico (ViT entrenado en ImageNet).
- Gemini "lee" toda esa información (incluido el texto en la imagen, los personajes, el contexto cultural) y la convierte en texto coherente.
- XLM-R fine-tuneado procesa ese texto rico mucho mejor que ViT procesando píxeles.
- En resumen: **delegamos la comprensión visual a Gemini y le damos a la red neuronal solo texto** — que es lo que XLM-R hace mejor.

A partir de aquí, todas las arquitecturas son derivadas de Vista E:
- **Vista E-2.2** (Task 2.2): Vista E + 7 features extra de Gemini (intention_probs, irony) + head jerárquico binario+tipo.
- **Vista E-2.3** (Task 2.3): Vista E + 6 features extra de Gemini (category_probs) + head_sex + 5 sigmoides condicionales.

---

**ETAPA 4 — Experimentos finales: ¿el cuello de botella es el `max_length`?**

Descubrimos que con `max_length=320` el 82% de los memes de Task 2.2 se truncaba (mediana del texto: 378 tokens). Eso significa que la red veía texto cortado en 4 de cada 5 ejemplos. Decidimos probar:

- **Variante max=512**: mismo XLM-R-base, pero `max_length` subido a 512 (su máximo nativo). Mismo input que Vista E, simplemente sin cortar.
- **Variante Longformer-4096**: cambiamos el backbone a `markussagen/xlm-roberta-longformer-base-4096`, que soporta hasta 4096 tokens con atención esparsa. `max_length` = 1100 (el máximo real observado en nuestros datos es 1041).
- **Variante con reasoning (_R)**: añadimos también el campo `reasoning` de Gemini al texto enriquecido, para ver si aporta señal extra.

**Resultados clave de la etapa 4:**
- En **2.1**: max=512+R sube ICM a **+0.411** (de +0.386). Mejora moderada.
- En **2.2**: el Longformer sube F1macro de **0.560 → 0.602** y F1 JUDG de **0.260 → 0.374** (+50%). **Salto enorme.** Confirma que el truncamiento era el cuello de botella.
- En **2.3**: max=512+sampler sube F1macro de **0.581 → 0.715** (+23%). Otro salto enorme — aquí el limitante era doble: truncamiento + dataset desbalanceado.

### C. Lo que entra exactamente a la red en CADA modelo del v4 entregado

#### Task 2.1 — los 3 runs hard / 3 runs soft

**hard_1: Vista E max=512 + reasoning (R) — el mejor en hard**
```
Texto = "{OCR} Description: {Gemini.description} Sexism Analysis: {Gemini.sexism_analysis} Reasoning: {Gemini.reasoning}"
        → tokens (max 512) → XLM-R fine-tune → mean-pool → 768
EEG     = matriz (n_sub, 80) → SetAttentionPool → 256
Ekman   = 7 dims softmax
─────────────────────────────────────────────────────
TOTAL: 768 + 256 + 7 = 1031 dims → MLP → sigmoid → P(YES)
```

**hard_2: Vista E Longformer — segundo mejor**
```
Texto = mismo enriquecido (sin reasoning)
        → tokens (max 1100) → Longformer-4096 → mean-pool → 768
EEG     = → 256
Ekman   = 7
─────────────────────────────────────────────────────
TOTAL: 1031 dims → MLP → sigmoid → P(YES)
```

**hard_3: Vista E max=512 (sin reasoning)** — mismo formato que hard_1 pero sin el `reasoning` en el texto.

**soft_1: blend Vista E + Gemini — el mejor en soft**
```
P_blend = 0.5 · P_VistaE_original  +  0.5 · P_Gemini_zero_shot
                    ↑ red entrenada              ↑ zero-shot, sin entrenar
                                                  (probabilidad sale directamente del JSON de Gemini)
```
**No es una sola red** — es una combinación: la salida es el promedio de dos modelos.

**soft_2: Vista E Longformer** — mismo modelo que hard_2.
**soft_3: Vista E Longformer + R** — Longformer con `reasoning` añadido al texto.

#### Task 2.2 — los 3 runs hard / 3 runs soft

**hard_1: Vista E-2.2 Longformer + thr — el mejor**
```
Texto = "{OCR} </s> {desc} </s> INTENTION: {intention_reasoning} </s> IRONY: {irony_s} </s> {sexism_analysis} </s> {reasoning}"
        → Longformer-4096 → mean-pool → 768
EEG     = → 256
Ekman   = 7
Gemini-feat-2.2 = [sexist_prob, confidence, P_NO, P_DIR, P_JUDG, irony_flag, irony_conf] = 7
─────────────────────────────────────────────────────
TOTAL: 768 + 256 + 7 + 7 = 1038 dims → trunk(1038→256) → HierarchicalHead
                                          ↓                  ↓
                                        bin_head(256→1)    type_head(256→2)
                                          ↓                  ↓
                                        P(sex) ─────────→ P(DIR|sex), P(JUDG|sex)
                                          ↓
                            P(NO)=1-P(sex), P(DIR)=P(sex)·P(DIR|sex), P(JUDG)=P(sex)·P(JUDG|sex)
```

**hard_2: Vista E-2.2 max=512 + thr** — mismo input que hard_1 pero con XLM-R-base y max=512.
**hard_3: blend Vista E-2.2 + Gemini** — combinación de Vista E-2.2 + Gemini.

**soft_1: blend Vista E-2.2 + Gemini RAW** — el blend sin calibrar.
**soft_2: Vista E-2.2 Longformer + Platt** — mismo modelo que hard_1, pero las probs calibradas con Platt scaling.
**soft_3: Vista E-2.2 max=512 + Platt** — mismo modelo que hard_2 + Platt.

#### Task 2.3 — los 3 runs hard / 3 runs soft

**hard_1: Vista E-2.3 max=512 + Sampler + thr — el mejor**
```
Texto = "{OCR} </s> {desc} </s> CATEGORIES: {gemini.categories_present} </s> {gemini.category_reasoning} </s> {gemini.sexism_analysis}"
        → tokens (max 512) → XLM-R-base fine-tune (con WeightedRandomSampler en train) → mean-pool → 768
EEG     = → 256
Ekman   = 7
Gemini-feat-2.3 = [sexist_prob, P_cat1, P_cat2, P_cat3, P_cat4, P_cat5] = 6
─────────────────────────────────────────────────────
TOTAL: 768 + 256 + 7 + 6 = 1037 dims → trunk(1037→256)
                                          ↓
                              head_sex(256→1)        head_cat(256→5)
                                  ↓                       ↓
                              P(sex)               P(cat|sex) para cada categoría
                                  ↓                       ↓
                              P_marg(cat_i) = P(sex) · P(cat_i|sex)
```

**hard_2: Vista E-2.3 max=512 sin Sampler** — mismo todo, sin el WeightedRandomSampler en el entrenamiento (afecta a los pesos del modelo, no a las features de entrada).
**hard_3: Vista E-2.3 max=512 + Sampler + Reasoning** — añade `reasoning` al texto enriquecido.

**soft_1: Vista E-2.3 Longformer sin Sampler — el mejor en soft**: cambia backbone a Longformer y quita el sampler (mejor calibración).
**soft_2: Vista E-2.3 max=512 sin Sampler** — XLM-R-base sin sampler.
**soft_3: Vista E-2.3 Longformer + Sampler** — Longformer con sampler.

---

### D. En una línea por subtarea

- **2.1** = "XLM-R-base lee `OCR + descripción Gemini + análisis Gemini + reasoning Gemini`, le concatenamos EEG y Ekman, y un MLP saca P(sexista)."
- **2.2** = "Idem 2.1 + Longformer (texto cabe sin truncar) + 7 features extra de Gemini sobre intención + head jerárquico de 2 ramas."
- **2.3** = "Idem 2.1 + max=512 + sampler en train + 6 features extra de Gemini sobre categorías + head con 5 sigmoides condicionales."

### E. Lo que NO entra a la red (y por qué)

- **La imagen "cruda" como píxeles**: ningún modelo final ve la imagen directamente. Solo a través de Gemini que la "describe" en texto.
- **El embedding del ViT**: solo se usa en M1/M2/M3 vistas A-D, NO en Vista E ni en sus derivados de 2.2/2.3.
- **ET y HR**: se incluyen en M1/M2 pero NO en Vista E ni en sus derivados — porque empíricamente no aportan suficiente vs la descripción de Gemini.
- **Los meta-datos del meme** (lang, anotadores, edad/género/país de los anotadores...): NO entran a la red. El idioma se gestiona pasivamente via XLM-RoBERTa que es multilingüe.

---

## 18. Lecciones aprendidas (extensión)

A las 10 lecciones de la sección 11, añadimos las que descubrimos hoy:

11. **Medir el truncamiento antes de elegir `max_length`**. Es un cinco minutos y puede revelar problemas estructurales. En 2.2 el 82% se truncaba con max=320 — eso era el techo, no la arquitectura.
12. **`max_length` < cuello de botella REAL del texto enriquecido** te limita más que el modelo. Subir de 320 → 512 en 2.2 subió F1 JUDG de 0.260 → 0.374 (+50%).
13. **Longformer no siempre gana**. Es mejor cuando el texto se trunca demasiado en el modelo base (Task 2.2). Pero cuando ya cabe bien (Task 2.3 con max=512), Longformer pierde ligeramente — porque su atención esparsa es menos óptima que la densa para textos cortos-medios.
14. **El `WeightedRandomSampler` es asimétrico:** mejora hard (boost minorías) y empeora soft (sobre-confianza). Por eso para los runs hard de 2.3 usamos sampler y para los soft, no.
15. **Texto enriquecido adicional (`reasoning`) NO siempre ayuda.** En 2.3 fue inútil — el `category_reasoning` ya cubre la información. En 2.1 sí aportó (+0.025 ICM hard). En 2.2 el efecto depende del backbone: con max=512 perjudica (ICM cae); con Longformer ayuda (porque cabe entero).
16. **El blend Vista E + Gemini sigue siendo imbatible en soft**. Ningún Vista E puro (ni siquiera Longformer fine-tuneado) supera el +0.596 ICMSoft de 2.1. El ensemble suaviza la sobre-confianza del modelo entrenado, y eso es exactamente lo que ICM-soft premia.
17. **Diversidad de modelos como póliza de seguro:** v3 (máxima diversidad) sacrifica algo de métrica para tener 3 paradigmas distintos por run. Si un backbone falla en el test, los otros compensan. v4 priorizó la métrica pero también logra cierta diversidad porque XLM-R-base, Longformer y blend cubren 3 puntos del espacio de soluciones.
18. **Bugs de PyEvALL son reales y silenciosos.** El bug de sigma=0 hacía que las métricas soft de 2.3 dieran NaN/crashaban sin error claro. Un monkeypatch de 5 líneas lo arregla.
