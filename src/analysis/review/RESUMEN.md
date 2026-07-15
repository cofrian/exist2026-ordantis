# RESUMEN — Análisis de revisión EXIST 2026 Task 2 (equipo Ordantis)

Todos los números provienen de **ejecutar código sobre datos reales** (validación, n=598;
split estratificado 85/15 por idioma y etiqueta, `SEED=42`). **No hay ninguna cifra inventada.**
Ningún modelo fue reentrenado: se reutilizan los 16 checkpoints entregados y la caché de Gemini.

- Entorno montado: venv en `../.venv` (torch 2.12.1+cu130 sobre RTX 4500 Ada, transformers 5.12.1,
  pyevall 0.2.11, scikit-learn 1.9, sentencepiece/protobuf para los tokenizers Longformer).
- Todos los logs crudos están en `resultados_revision/logs/`.

---

## TAREA 0 — Verificación (OK)
`config.py` resuelve todas las rutas por glob:
- `TRAIN_JSON` y `TEST_JSON` existen; `CKPT_DIR` (3 ckpt) + `_alt/` (13 ckpt) = **16 checkpoints**.
- Caché Gemini en `preprocessed/gemini_predictions.json` (+ `.DONE`), Ekman, sensor_stats, ViT emb. OK.
- `DEVICE=cuda`. Split real: train=3386, val=598, test=1053.

---

## TAREA 1 — Métricas reales (validación) → `metricas_18_modelos.csv` (28 filas)

Métrica primaria por subtarea: 2.1 = F1(positiva), 2.2/2.3 = F1 macro. ICM/ICMSoft vía PyEvALL
(con el parche `sigma=0` que ya traía el código). Todas sobre validación.

### 2.1 — sexista sí/no (binario)
| Modelo | F1+ | AUC | ICM | ICMSoft |
|---|---|---|---|---|
| Vista E (M3_vista_E, el del zip) | 0.8609 | 0.8839 | +0.3855 | +0.4798 |
| Gemini 3-flash crudo | 0.8748 | 0.8550 | +0.3429 | +0.2335 |
| Ensemble 0.6·E + 0.4·Gemini | 0.8560 | 0.8886 | +0.3733 | **+0.5957** |
| Vista E-2.1 max512 | 0.8715 | 0.8852 | +0.3944 | +0.3025 |
| Vista E-2.1 max512_R | 0.8633 | 0.8925 | **+0.4088** | +0.3878 |
| Vista E-2.1 Longformer | 0.8676 | 0.8833 | +0.4008 | +0.5428 |
| Vista E-2.1 Longformer_R | **0.8790** | 0.8843 | +0.3801 | +0.5334 |

### 2.2 — intención (NO / DIRECT / JUDGEMENTAL)
Checkpoint principal (`vista_e_task22_best.pt`), varias decodificaciones (log `t1_task22_reeval.log`):
mejor **ICM=+0.0349** con *VistaE + threshold ponderado* (F1macro=0.5597) y *blend+thr* (F1macro=0.5868);
argmax cae a ICM=−0.13. En soft, el mejor **ICMSoft=−0.157** es *blend 0.6E+0.4G raw*.
Variantes `_alt` (F1macro con threshold 2D óptimo):
| Modelo | F1macro | ICM | ICMSoft |
|---|---|---|---|
| Vista E-2.2 max512 | 0.5456 | −0.0645 | −0.6858 |
| Vista E-2.2 max512_R | 0.5669 | −0.0002 | −0.8021 |
| Vista E-2.2 Longformer | **0.6073** | **+0.1129** | −0.6758 |

### 2.3 — categorización (5 categorías multilabel)
| Modelo | F1micro | F1macro | ICM | ICMSoft |
|---|---|---|---|---|
| Vista E-2.3 ORIGINAL (zip) † | 0.5496 | 0.6741 | +0.2623 | −2.7553 |
| Vista E-2.3 max512 | 0.6238 | 0.7137 | +0.3363 | −4.5932 |
| Vista E-2.3 max512_v2 | 0.6251 | **0.7146** | **+0.3417** | −5.9912 |
| Vista E-2.3 max512_R | 0.6149 | 0.7135 | +0.3331 | −6.1419 |
| Vista E-2.3 Longformer | 0.5856 | 0.7051 | +0.2647 | −3.4947 |
| Vista E-2.3 Longformer_v2 | 0.6188 | 0.7029 | +0.2455 | −5.1343 |
| Vista E-2.3 Longformer_R | 0.6144 | 0.7116 | +0.3067 | −5.4749 |
| Gemini 3-flash crudo (2.3) | 0.5938 | 0.5367 | −3.4148 | −17.1151 |

† El checkpoint **principal 2.3** (el del zip) tiene una **interfaz distinta** a las variantes:
`VistaE23` emite solo 5 sigmoides de categoría (sin cabeza de "sexista"), por lo que su `infer`
devuelve 4 valores y el harness genérico `_full_eval_task23.py` (escrito para las variantes de 6
salidas) no lo desempaqueta. Se evaluó con un **evaluador dedicado** (`eval_main23.py`) usando una
compuerta de sexista = máx. prob. de categoría. Además `_full_eval_task23.py` llamaba
`task23.load_t23()` cuando la función real es `load_task23()` (se corrigió con un alias en runtime,
`run_full_eval_task23_fixed.py`, sin editar el original).

---

## TAREA 2 — Calibración → `calibracion.csv` + reliability diagrams

ECE (10 bins), MCE, Brier. Gold **hard** = voto mayoritario (umbral 0.5); reliability **soft** =
prob. predicha vs proporción real de anotadores. Figuras: `figuras/reliability_2_1.png`,
`reliability_2_2_perclass.png`, `reliability_2_3_percat.png`.

**Antes vs después de calibrar (ECE-hard):**
| Subtarea | Método | ECE (raw) | ECE (calibrado) |
|---|---|---|---|
| 2.1 (Vista E) | temperature (T=1.149) | 0.108 | 0.133 (empeora hard; el soft-ECE mejora 0.029→0.019) |
| 2.1 (Vista E→Ensemble) | blend 0.6/0.4 con Gemini | 0.108 | **0.075** |
| 2.2 (VistaE22, macro OvR) | Platt por clase | 0.131 | **0.036** |
| 2.3 (VistaE23, macro OvR) | Platt por clase | 0.094 | **0.035** |

- **Platt por clase mejora mucho el ECE-hard** en 2.2 y 2.3 (a costa de subir a veces el soft-ECE,
  porque se ajusta a etiquetas duras). Peor categoría cruda en 2.2: JUDGEMENTAL (ECE=0.178 → 0.016).
- El **temperature scaling** de 2.1 casi no ayuda al hard (T≈1.15, el modelo ya está casi calibrado);
  el **ensemble con Gemini** sí baja el ECE-hard.
- **Caveat honesto:** Platt/temperature se ajustan *in-sample* sobre validación (no hay otro conjunto
  con gold), así que la mejora es una **cota superior** del beneficio real.

---

## TAREA 3 — Análisis de errores 2.3 vs gold → `errores_2_3.csv`, `errores_2_3_resumen.md`, `figuras/confusion_2_3.png`

Modelo principal, validación. Umbrales óptimos sobre val: `thr_sex=0.34`, `thr_cat=0.20`.

| Categoría | freq_gold | F1 | Precisión | Recall | FP | FN |
|---|---|---|---|---|---|---|
| SEXUAL-VIOLENCE | 153 | 0.676 | 0.602 | 0.771 | 78 | 35 |
| MISOGYNY-NON-SEXUAL-VIOLENCE | 186 | **0.556** | 0.488 | 0.645 | 126 | 66 |
| IDEOLOGICAL-INEQUALITY | 239 | 0.688 | 0.622 | 0.770 | 112 | 55 |
| OBJECTIFICATION | 262 | 0.712 | 0.707 | 0.718 | 78 | 74 |
| STEREOTYPING-DOMINANCE | 301 | 0.712 | 0.698 | 0.728 | 95 | 82 |

- **Correlación frecuencia↔F1: r=+0.587** (moderada): la rareza explica *en parte* el F1, pero **no todo**.
- **MISOGYNY-NON-SEXUAL-VIOLENCE es el outlier**: peor F1 (0.556) **sin ser la más rara** → es
  **confusión de frontera**, no falta de datos. Las off-diagonales de `P(pred cat | gold cat)` lo
  confirman: memes con gold MISOGYNY-NSV se predicen también como STEREOTYPING-DOMINANCE (0.71) e
  IDEOLOGICAL-INEQUALITY (0.66). Otras confusiones fuertes: SEXUAL-VIOLENCE↔OBJECTIFICATION (0.75),
  IDEOLOGICAL↔STEREOTYPING (0.73). Coherente con la alta co-ocurrencia real en gold
  (`cooc_gold_2_3.csv`: IDEOLOGICAL & STEREOTYPING co-ocurren en 202 memes).

---

## TAREA 4 — Ablación del aporte de Gemini → `ablacion_gemini_features.csv`

| Subtarea | Condición | F1macro | ICM | ICMSoft |
|---|---|---|---|---|
| 2.2 | normal (con 7 gfeat de Gemini) | 0.5585 | +0.0086 | −0.6787 |
| 2.2 | **ablación (gfeat = 0)** | 0.5398 | −0.0243 | −0.7963 |
| 2.3 | normal (texto con Gemini) | 0.6741 | +0.2623 | −2.7553 |
| 2.3 | **ablación (solo OCR, sin texto Gemini)** | **0.5151** | **−1.2259** | −6.1976 |

- **2.2:** las 7 features numéricas de Gemini (`sexist_prob, confidence, P_NO, P_DIRECT, P_JUDG,
  irony_flag, irony_conf`) aportan **poco pero positivo**: quitarlas baja F1macro −0.019 y ICM −0.033.
- **2.3 (aclaración importante):** `VistaE23` **no consume ninguna feature numérica de Gemini** en el
  `forward` (solo texto 768 + EEG 256 + Ekman 7). Poner "6 features a cero" es **N/A**. La ablación
  equivalente y honesta es quitar el **texto** derivado de Gemini: al dejar solo el OCR, F1macro cae
  **0.674 → 0.515** e ICM **+0.262 → −1.226**. → En 2.3 **el texto de Gemini es el aporte dominante**.
- (Las 11 "disagreement features" de 2.3 son *salida* de una tarea auxiliar, no entrada del modelo.)

---

## TAREA 5 — Parámetros por checkpoint → `parametros.csv`

16 checkpoints, contados desde el `model_state_dict` (backbone incluido; se excluyen buffers
`position_ids`/`token_type_ids` para igualar `sum(p.numel())`).

- Familias XLM-R base: **~278.4–278.8 M** parámetros (backbone ~278.0 M; cabezas 0.38–0.75 M).
- Familias Longformer: **~281.4–281.5 M** (backbone ~280.8 M).
- Rango total: 278,428,166 – 281,547,271 parámetros. El coste está dominado por el backbone de texto;
  las cabezas jerárquica (2.2) / multilabel+aux (2.3) añaden < 0.8 M.

---

## Ficheros generados (`resultados_revision/`)
- Tablas: `metricas_18_modelos.csv`, `calibracion.csv`, `errores_2_3.csv`,
  `cooc_gold_2_3.csv`, `cooc_pred_2_3.csv`, `confus_goldcat_predcat_2_3.csv`,
  `ablacion_gemini_features.csv`, `parametros.csv`, `task21_variants.csv`, `task22_variants.csv`.
- Figuras: `figuras/reliability_2_1.png`, `figuras/reliability_2_2_perclass.png`,
  `figuras/reliability_2_3_percat.png`, `figuras/confusion_2_3.png`.
- Texto: `errores_2_3_resumen.md`, este `RESUMEN.md`.
- Scripts nuevos (no editan los originales): `eval_main23.py`, `run_full_eval_task23_fixed.py`,
  `eval_task21_variants.py`, `eval_task22_variants.py`, `consolidate_metrics.py`,
  `tarea2_calibracion.py`, `tarea3_errores_23.py`, `tarea4_ablacion.py`, `tarea5_parametros.py`.
- Logs crudos de cada ejecución en `logs/`.

---

## Limitaciones (explícitas)
1. **Todas las métricas son sobre VALIDACIÓN (n=598), nunca sobre TEST.** El test de EXIST **no tiene
   gold público**, así que cualquier ICM/F1/calibración sobre test sería imposible sin inventar. Las
   predicciones de test se generan (submissions/zip) pero **no se evalúan**.
2. **Calibración in-sample:** Platt y temperature se ajustan sobre la misma validación donde se mide
   el ECE (no hay un tercer conjunto con gold). Las mejoras de ECE son cotas superiores optimistas.
3. **Gold "hard" = voto mayoritario a 0.5**; en 2.3 una categoría se considera presente si la marca
   > 1/6 de los anotadores (coherente con `gold_hard_from_soft` del código original). Otras
   binarizaciones darían cifras algo distintas.
4. **El modelo 2.3 principal no tiene cabeza de "sexista"**; su ICM se calculó con una compuerta
   sexista = máx. prob. de categoría (decodificación documentada en `eval_main23.py`), no con una
   probabilidad de sexismo entrenada. Por eso su ICMSoft no es 1:1 comparable con las variantes de 6
   salidas.
5. **Ablación 2.3 = ablación de TEXTO** (no de features numéricas, que no existen en ese modelo).
6. Se corrigieron dos bugs del harness de evaluación (nombre `load_t23` vs `load_task23`, y tokenizers
   Longformer que requerían `sentencepiece`+`protobuf`) **sin reentrenar ni editar los scripts
   originales** (alias en runtime + dependencias instaladas).
