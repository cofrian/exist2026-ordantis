# README — Trazabilidad del análisis completo (16 checkpoints)

**Fecha de ejecución:** 2026-07-03, ~20:05 (hora local).
**Proyecto:** EXIST 2026 Task 2 (memes sexistas), equipo *Ordantis*.

## Entorno
| Componente | Versión |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| Transformers | 5.12.1 |
| scikit-learn | 1.9.0 |
| SciPy | 1.17.1 |
| PyEvALL | 0.2.11 |
| GPU | NVIDIA RTX 4500 Ada |

## Datos base
- **Split de validación:** partición estratificada 85/15 por idioma y etiqueta binarizada, `SEED=42`.
  Tamaños confirmados en ejecución: **train=3386, val=598, test=1053**. ✅
- **Caché Gemini:** `exist2026_Ordantis/preprocessed/gemini_predictions.json` (usada tal cual, sin regenerar).
- **Gold de validación:** derivado de las anotaciones de `EXIST2026_training.json` filtradas a los 598 IDs de validación.
- **PyEvALL** con parche `sigma=0` (idéntico al del código original `_full_eval_task23.py`) para evitar `StatisticsError` cuando la desviación gold es 0.

## Umbrales de gold aplicados (según instrucción)
| Subtarea | Gold hard | Gold soft |
|---|---|---|
| 2.1 | nº anotadores YES > 3 | proporción YES |
| 2.2 | clase mayoritaria (argmax de votos) | proporción por clase |
| 2.3 | nº anotadores por categoría > 1 (= proporción > 1/6) | proporción por categoría; sexismo = proporción de anotadores que marcan ≥1 categoría |

- **Calibración (Bloque 1) — binarización del gold por subtarea:**
  - **2.1:** umbral **0.5** (mayoría de anotadores YES).
  - **2.2:** clase **mayoritaria** (argmax de votos), OvR por clase.
  - **2.3:** cada categoría a **1/6** (`bloque1_calibracion.py` líneas 112/123), **el mismo umbral que
    la evaluación oficial** de Bloques 2/5. Además el soft de categorías es idéntico entre el gold de
    `_common` y el oficial (Δ=0), y como el gold oficial trata los 598 como sexistas (SX=1) no hay
    compuerta de sexismo que aplicar → **la calibración de 2.3 es consistente con la evaluación
    oficial** (verificado: re-ejecutar da CSV byte-idéntico).
- **Errores 2.3 (Bloque 2) y métricas 2.3 (Bloque 5) — CORREGIDO 2026-07-04:** se usa la
  **decodificación oficial de `_full_eval_task23.py`**, la que reproduce Table 6 del paper:
  - **Compuerta de sexismo = cabeza de sexismo REAL** (6ª salida) de cada una de las 6 variantes.
    Para el principal `vista_e_task23_best` (sin esa cabeza) se aproxima con `max(P_categoría)`.
  - **Umbral por búsqueda oficial `find_best_thr`** (maximiza `0.5·Fmacro + 0.5·ICMNorm`;
    tsex∈[0.30,0.66), tcat uniforme∈[0.05,0.55)). Para las 6 variantes el óptimo es **tsex=0.30,
    tcat=0.15**; para el principal, tsex=0.30/tcat=0.20.
  - **Gold oficial (Módulo 1):** categoría positiva si **>1 anotador** (`soft > 1/6`), con los 598
    memes tratados como categorizables (SX=1). Es el gold que reproduce el paper y **el mismo que se
    usa en TODOS los descriptivos de 2.3** (`cooc_gold_2_3.csv`, `densidad_multietiqueta_gold.csv`,
    `gemini_prob_media_por_categoria.csv` freq_gold) — sus frecuencias por categoría coinciden con los
    F1 (346/372/324/183/210). Ya **no** se usa el gold realista de `_common` en ningún descriptivo.
  - **Verificación:** reproduce Table 6 → `max512_v2` = F1-macro 0.7146 / ICM +0.3417 (run 2.3 hard_1),
    `max512` 0.7137/+0.3363, `max512_R` 0.7135/+0.3331, etc.

## Modelos usados (los confirmados por byte-match en la sesión previa)
Los 16 checkpoints entregados: 5 de 2.1, 4 de 2.2, 7 de 2.3. Se usó la interfaz/configuración de cada uno
(max_len, captions, features) con la que se generaron las predicciones reales de la submission.

## Ficheros generados
| Bloque | Salida |
|---|---|
| 0 | `preds_val_<checkpoint>.csv` × 16 (probabilidades crudas) |
| 1 | `calibracion_todos.csv` (90 filas), `figuras/reliability_2_1.png`, `reliability_2_2_perclass.png`, `reliability_2_3_percat.png` |
| 2 | `errores_2_3_por_checkpoint.csv`, `confusion_por_checkpoint.csv`, `cooc_gold_2_3.csv`, `densidad_multietiqueta_gold.csv`, `gemini_prob_media_por_categoria.csv`, `figuras/confusion_2_3.png` |
| 3 | `ablacion_gemini_todos.csv` |
| 4 | `parametros_todos.csv` |
| 5 | `metricas_todos_checkpoints_val.csv` |
| 6 | `ablacion_fisiologia_principales.csv` |

## Corrección 2026-07-04 — Task 2.3 (Bloques 2 y 5)
La primera versión evaluó 2.3 con **compuerta `max(P_categoría)` para todos** los checkpoints y
umbral fijo tsex=0.30/tcat=0.15, dando **ICM negativos en todos** (no reproducía el paper). Se
corrigió usando la **cabeza de sexismo real** de las 6 variantes + `find_best_thr` + gold oficial.
Resultado: **se reproduce Table 6** (max512_v2 = 0.7146/+0.3417).

**CSVs regenerados:** `metricas_todos_checkpoints_val.csv` (solo filas 2.3; 2.1/2.2 intactas),
`errores_2_3_por_checkpoint.csv`, `confusion_por_checkpoint.csv`.
**También regenerados (consistencia con gold oficial):** `calibracion_todos.csv` (la calibración de
2.3 pasa a reportarse sobre `max512_v2`, el run entregado, no `task23_best`; `reliability_2_3_percat.png`
regenerada), `cooc_gold_2_3.csv`, `densidad_multietiqueta_gold.csv`, `gemini_prob_media_por_categoria.csv`
(freq_gold → gold oficial Módulo 1).
**Bloques 3 y 6 (parte 2.3) recalculados con gate real + gold oficial** (`«con» reproduce Bloque 5`):
`ablacion_gemini_todos.csv` (filas 2.3; 2.2 intactas) y `ablacion_fisiologia_principales.csv` (fila 2.3;
2.1/2.2 intactas). **Cambian conclusiones:** las features numéricas de Gemini en 2.3 aportan poco pero
consistente (antes «inútiles»); la fisiología **perjudica 2.3** además de 2.2 (antes «ayuda leve»).
**Intactos:** Bloque 4 (parámetros), Bloque 3 (ablación Gemini **2.2**), Bloque 6 (fisiología **2.1/2.2**),
predicciones base Bloque 0, calibración de 2.1/2.2.
**Scripts nuevos:** `_reinfer_23_gate.py`, `_recompute_23.py`, `_recuperar_thresholds_23.py`,
`_recompute_cooc_densidad.py`, `_recompute_bloque3_23.py`, `_recompute_bloque6_23.py`, `_pyevall23.py`;
**cache:** `cache_23_gate.npz`. Ver `thresholds_recuperados.md`.

## Incidencias / caveats
1. **Calibración in-sample:** Platt y temperature se ajustan sobre la MISMA validación donde se mide el ECE. Las mejoras de ECE son **cotas superiores optimistas**.
2. **Modelo 2.3 principal (`vista_e_task23_best`) sin cabeza de sexismo:** su compuerta se aproxima con `max(P_categoría)`; **el harness oficial no puede evaluarlo** (devuelve 4 valores, no 5). Bajo gold oficial puntúa mal en 2.3 hard (ICM −2.15). **No es el run entregado.** El run 2.3 hard_1 del paper es `vista_e_task23_max512_v2`.
3. **Ablación Gemini en 2.3:** solo aplica a las **6 variantes** (consumen 6 features numéricas vía la clave `feat`). El principal `vista_e_task23_best` **no consume features numéricas** de Gemini → celda marcada **NA** (no forzada).
4. **Gold 2.3 unificado:** todo el análisis de 2.3 (métricas, F1 por categoría, co-ocurrencia, densidad, freq_gold, y la binarización 1/6 de la calibración) usa **el mismo gold oficial (Módulo 1)**. El gold realista de `_common` ya no aparece en ningún descriptivo publicable.
5. Ninguna cifra ha sido estimada, extrapolada ni inventada. No hay celdas con NaN/inf en los CSVs finales.
