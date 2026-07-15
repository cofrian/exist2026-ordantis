# Informe de análisis para respuesta a revisores
## EXIST 2026 — Task 2 (clasificación de memes sexistas) — Equipo *Ordantis*

> **Ámbito.** Este documento recoge la batería completa de análisis solicitados por los
> revisores. Todas las cifras se han obtenido **ejecutando código sobre datos reales**, en
> **validación** (n = 598), reutilizando los **16 checkpoints** entregados y la caché de
> Gemini. **No se ha reentrenado ningún modelo** y **no hay ninguna cifra estimada**. Las
> métricas oficiales (ICM / ICM-Soft) se calculan con **PyEvALL 0.2.11**.

**Entorno de cómputo.** GPU NVIDIA RTX 4500 Ada; PyTorch 2.12.1 (CUDA 13.0); Transformers
5.12.1; scikit-learn 1.9; PyEvALL 0.2.11. Partición estratificada 85/15 por idioma y etiqueta
binaria (`SEED = 42`): **train = 3386, validación = 598, test = 1053**.

---

## 0. Resumen ejecutivo

| Subtarea | Mejor modelo (criterio) | Métrica principal | ICM | ICM-Soft |
|---|---|---|---|---|
| **2.1** sexista sí/no | Ensemble 0.6·VistaE + 0.4·Gemini | F1⁺ = 0.856 | +0.373 | **+0.596** |
| **2.2** intención | Vista E-2.2 Longformer | F1-macro = 0.607 | **+0.113** | −0.676 |
| **2.3** categorización | Vista E-2.3 max512_v2 | F1-macro = **0.715** | +0.342 | −5.991 |

**Cuatro conclusiones de una línea:**

1. **La subtarea 2.1 está resuelta y bien calibrada** (F1⁺ ≈ 0.86–0.88 en todas las variantes;
   el ensemble con Gemini maximiza el ICM-Soft).
2. **La subtarea 2.2 es la difícil**: el ICM ronda 0 y la clase **JUDGEMENTAL** hunde el macro.
3. **Gemini aporta poco como *features* numéricas (2.2) pero mucho como *texto* (2.3)**: retirar
   el texto de Gemini en 2.3 desploma F1-macro de 0.674 a 0.515.
4. **El peor F1 de 2.3 se debe a confusión de frontera, no a rareza**: MISOGYNY-NSV se confunde
   con STEREOTYPING e IDEOLOGICAL, categorías semánticamente contiguas.

---

## 1. Rendimiento de todos los modelos (Tarea 1)

Se evaluaron 28 configuraciones en validación: los 16 checkpoints, Gemini crudo en cada
subtarea, los ensembles/blends y las estrategias de decodificación de 2.2. La métrica de F1 se
adapta a la naturaleza de cada subtarea (binaria, multiclase, multietiqueta). El *threshold* se
optimiza sobre ICM (2.1) o sobre F1-macro con rejilla 2-D (2.2, 2.3).

### 1.1 Subtarea 2.1 — sexista sí / no (clasificación binaria)

| Modelo | F1⁺ | AUC-ROC | ICM | ICM-Soft | thr |
|---|--:|--:|--:|--:|--:|
| Vista E (M3_vista_E, *zip*) | 0.8609 | 0.8839 | +0.3855 | +0.4798 | 0.55 |
| Gemini 3-flash crudo | 0.8748 | 0.8550 | +0.3429 | +0.2335 | 0.30 |
| **Ensemble 0.6·E + 0.4·Gemini** | 0.8560 | 0.8886 | +0.3733 | **+0.5957** | 0.67 |
| Vista E-2.1 max512 | 0.8715 | 0.8852 | +0.3944 | +0.3025 | 0.60 |
| Vista E-2.1 max512_R | 0.8633 | 0.8925 | **+0.4088** | +0.3878 | 0.61 |
| Vista E-2.1 Longformer | 0.8676 | 0.8833 | +0.4008 | +0.5428 | 0.57 |
| **Vista E-2.1 Longformer_R** | **0.8790** | 0.8843 | +0.3801 | +0.5334 | 0.50 |

**Lectura.** El rendimiento es notablemente homogéneo (AUC ≈ 0.88–0.89). Ninguna variante de
mayor longitud de contexto (max512, Longformer) ni el añadido de *reasoning* (sufijo `_R`)
mejora de forma decisiva sobre el modelo del *zip*. El **ensemble con Gemini** es la
configuración con mejor calidad probabilística (ICM-Soft = +0.596), mientras que Gemini crudo,
pese a un F1⁺ ligeramente superior, es el más pobre en soft (+0.234): sus probabilidades están
peor calibradas (véase §2).

### 1.2 Subtarea 2.2 — intención: NO / DIRECT / JUDGEMENTAL

Checkpoint principal (`vista_e_task22_best`), evaluado bajo diez decodificaciones distintas:

| Estrategia | F1-macro | ICM | ICM-Norm | ICM-Soft | ICM-SoftNorm | F1 [NO/DIR/JUD] |
|---|--:|--:|--:|--:|--:|---|
| HARD · VistaE + *threshold* ponderado | 0.5597 | **+0.0349** | 0.513 | — | — | 0.78 / 0.65 / 0.25 |
| HARD · VistaE argmax | 0.5591 | −0.1302 | 0.452 | — | — | 0.76 / 0.56 / 0.36 |
| **HARD · blend 0.6E+0.4G + thr** | **0.5868** | +0.0349 | 0.513 | — | — | 0.73 / 0.69 / 0.34 |
| HARD · blend 0.6E+0.4G argmax | 0.5594 | −0.0447 | 0.484 | — | — | 0.67 / 0.67 / 0.33 |
| HARD · Gemini crudo argmax | 0.5497 | −0.0786 | 0.471 | — | — | 0.65 / 0.67 / 0.33 |
| SOFT · VistaE + Platt por clase | 0.5059 | — | — | −0.4388 | 0.454 | CE = 1.492 |
| SOFT · VistaE raw | 0.5591 | — | — | −0.6787 | 0.429 | CE = 1.380 |
| SOFT · blend + Platt por clase | 0.5816 | — | — | −0.2646 | 0.472 | CE = 1.485 |
| **SOFT · blend 0.6E+0.4G raw** | 0.5594 | — | — | **−0.1571** | 0.484 | CE = 1.399 |
| SOFT · Gemini crudo | 0.5497 | — | — | −2.5618 | 0.232 | CE = 2.045 |

Variantes alternativas (`_alt`), con *threshold* 2-D óptimo:

| Modelo | F1-macro (thr) | F1-macro (argmax) | ICM | ICM-Norm | ICM-Soft | F1 [NO/DIR/JUD] |
|---|--:|--:|--:|--:|--:|---|
| Vista E-2.2 max512 | 0.5456 | 0.5752 | −0.0645 | 0.476 | −0.6858 | 0.77 / 0.58 / 0.38 |
| Vista E-2.2 max512_R | 0.5669 | 0.5697 | −0.0002 | 0.500 | −0.8021 | 0.76 / 0.56 / 0.38 |
| **Vista E-2.2 Longformer** | **0.6073** | 0.5795 | **+0.1129** | 0.541 | −0.6758 | 0.75 / 0.63 / 0.36 |

**Lectura.** Es la subtarea más difícil: el ICM se sitúa alrededor de 0, lo que indica que el
modelo apenas supera el *baseline* jerárquico de PyEvALL. Dos patrones claros: (i) el
*threshold* ponderado mejora sistemáticamente al `argmax` (que castiga el ICM por sobre-predecir
la clase mayoritaria), y (ii) el *blend* con Gemini mejora la calidad soft. La clase
**JUDGEMENTAL** es el cuello de botella persistente (F1 entre 0.22 y 0.38) pese al
sobre-muestreo y a la *focal loss*. El único modelo con ICM claramente positivo es la variante
**Longformer** (+0.113).

### 1.3 Subtarea 2.3 — categorización (5 categorías, multietiqueta)

| Modelo | F1-micro | F1-macro | ICM | ICM-Norm | ICM-Soft | ICM-SoftNorm |
|---|--:|--:|--:|--:|--:|--:|
| Vista E-2.3 ORIGINAL (*zip*) | 0.5496 | 0.6741 | +0.2623 | 0.556 | −2.7553 | 0.355 |
| Vista E-2.3 max512 | 0.6238 | 0.7137 | +0.3363 | 0.570 | −4.5932 | 0.210 |
| **Vista E-2.3 max512_v2** | 0.6251 | **0.7146** | **+0.3417** | 0.571 | −5.9912 | 0.121 |
| Vista E-2.3 max512_R | 0.6149 | 0.7135 | +0.3331 | 0.569 | −6.1419 | 0.112 |
| Vista E-2.3 Longformer | 0.5856 | 0.7051 | +0.2647 | 0.555 | −3.4947 | 0.279 |
| Vista E-2.3 Longformer_v2 | 0.6188 | 0.7029 | +0.2455 | 0.551 | −5.1343 | 0.175 |
| Vista E-2.3 Longformer_R | 0.6144 | 0.7116 | +0.3067 | 0.564 | −5.4749 | 0.154 |
| Gemini 3-flash crudo (2.3) | 0.5938 | 0.5367 | −3.4148 | 0.000 | −17.1151 | 0.000 |

**Lectura.** En F1-macro las variantes `max512` / `_v2` / `_R` (≈ 0.71) superan al modelo del
*zip* (0.674). Todos los modelos entrenados baten con holgura a Gemini crudo (ICM +0.34 frente a
−3.41). El ICM-Soft es negativo en todos los casos porque la métrica soft penaliza con dureza la
sobre-confianza en las categorías raras; de ahí la importancia del análisis de calibración (§2).

> **Nota metodológica.** El checkpoint principal de 2.3 (`vista_e_task23_best`, el del *zip*)
> usa una arquitectura distinta a las variantes: emite únicamente 5 sigmoides de categoría, sin
> cabeza de "sexista". Se evaluó con un evaluador dedicado (`eval_main23.py`) empleando una
> **compuerta de sexista = máxima probabilidad de categoría**. Por ello su ICM-Soft no es
> directamente comparable con las variantes de 6 salidas.

---

## 2. Calibración (Tarea 2)

Se calculan **ECE** (Expected Calibration Error, 10 bins), **MCE** (Maximum Calibration Error) y
**Brier score**. El gold *hard* es el voto mayoritario (umbral 0.5); la fiabilidad *soft*
compara la probabilidad predicha con la proporción real de anotadores. Se compara **antes vs
después de calibrar**: *temperature scaling* y *blend* con Gemini en 2.1; **Platt por clase** en
2.2 y 2.3. Curvas de fiabilidad: `figuras/reliability_2_1.png`,
`figuras/reliability_2_2_perclass.png`, `figuras/reliability_2_3_percat.png`.

### 2.1 Resumen antes → después (ECE-hard; menor es mejor)

| Subtarea · método | ECE (raw) | ECE (calibrado) | Δ |
|---|--:|--:|---|
| 2.1 Vista E · *temperature* (T = 1.149) | 0.1084 | 0.1330 | empeora hard* |
| 2.1 Vista E → Ensemble · blend 0.6/0.4 | 0.1084 | **0.0748** | −0.034 |
| 2.2 VistaE22 (macro OvR) · Platt por clase | 0.1314 | **0.0361** | −0.095 |
| 2.3 VistaE23 (macro OvR) · Platt por clase | 0.0944 | **0.0347** | −0.060 |

\* El *temperature* empeora el ECE-hard (el modelo ya está casi calibrado, T ≈ 1.15) pero mejora
el soft-ECE (0.0294 → 0.0190).

### 2.2 Detalle 2.1 — calibración global (binario)

| Modelo · método | ECE-hard | MCE-hard | Brier-hard | ECE-soft | Brier-soft |
|---|--:|--:|--:|--:|--:|
| Vista E · raw | 0.1084 | 0.2316 | 0.1397 | 0.0294 | 0.0481 |
| Vista E · temperature | 0.1330 | 0.2530 | 0.1451 | 0.0190 | 0.0477 |
| Gemini crudo · raw | 0.0814 | 0.5500 | 0.1438 | 0.1996 | 0.0915 |
| **Ensemble 0.6E+0.4G · blend** | 0.0748 | 0.2869 | 0.1287 | 0.0858 | 0.0528 |

### 2.3 Detalle 2.2 — ECE por clase (one-vs-rest), raw vs Platt

| Clase | ECE raw | ECE Platt | MCE raw | MCE Platt | Brier raw | Brier Platt |
|---|--:|--:|--:|--:|--:|--:|
| NO | 0.0968 | 0.0396 | 0.2371 | 0.1163 | 0.1736 | 0.1604 |
| DIRECT | 0.1190 | 0.0524 | 0.3620 | 0.1105 | 0.1851 | 0.1628 |
| JUDGEMENTAL | 0.1782 | 0.0164 | 0.3845 | 0.4361 | 0.1238 | 0.0892 |
| **MACRO OvR** | **0.1314** | **0.0361** | — | — | — | — |

### 2.4 Detalle 2.3 — ECE por categoría (one-vs-rest), raw vs Platt

| Categoría | ECE raw | ECE Platt | MCE raw | MCE Platt | Brier raw | Brier Platt |
|---|--:|--:|--:|--:|--:|--:|
| IDEOLOGICAL-INEQUALITY | 0.0828 | 0.0430 | 0.2039 | 0.1829 | 0.1260 | 0.1206 |
| STEREOTYPING-DOMINANCE | 0.0669 | 0.0337 | 0.1885 | 0.1238 | 0.1694 | 0.1631 |
| OBJECTIFICATION | 0.1004 | 0.0440 | 0.2062 | 0.2299 | 0.1364 | 0.1255 |
| SEXUAL-VIOLENCE | 0.1184 | 0.0279 | 0.1874 | 0.2574 | 0.0785 | 0.0666 |
| MISOGYNY-NON-SEXUAL-VIOLENCE | 0.1032 | 0.0247 | 0.4459 | 0.4548 | 0.0925 | 0.0826 |
| **MACRO OvR** | **0.0944** | **0.0347** | — | — | — | — |

**Lectura.** El **Platt por clase reduce el ECE-hard alrededor de un 70 %** tanto en 2.2 como en
2.3; la categoría peor calibrada en crudo es JUDGEMENTAL (ECE 0.178 → 0.016). El coste es que en
ocasiones sube el soft-ECE, porque Platt se ajusta a etiquetas duras. En 2.1 el modelo ya está
casi calibrado (T ≈ 1.15) y es el *blend* con Gemini lo que reduce el ECE-hard. **Recomendación
para el paper:** reportar ECE por clase (no solo global) y aplicar Platt como post-proceso en
2.2/2.3.

---

## 3. Análisis de errores de la categorización 2.3 (Tarea 3)

Modelo principal, validación, umbrales óptimos `thr_sex = 0.34`, `thr_cat = 0.20`. Al ser
multietiqueta, se reduce a una confusión binaria por categoría.

### 3.1 Rendimiento por categoría (orden por frecuencia gold)

| Categoría | freq gold | freq pred | F1 | Precisión | Recall | TP | FP | FN | TN |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SEXUAL-VIOLENCE | 153 | 196 | 0.6762 | 0.6020 | 0.7712 | 118 | 78 | 35 | 367 |
| **MISOGYNY-NON-SEXUAL-VIOLENCE** | 186 | 246 | **0.5556** | 0.4878 | 0.6452 | 120 | 126 | 66 | 286 |
| IDEOLOGICAL-INEQUALITY | 239 | 296 | 0.6879 | 0.6216 | 0.7699 | 184 | 112 | 55 | 247 |
| OBJECTIFICATION | 262 | 266 | 0.7121 | 0.7068 | 0.7176 | 188 | 78 | 74 | 258 |
| STEREOTYPING-DOMINANCE | 301 | 314 | 0.7122 | 0.6975 | 0.7276 | 219 | 95 | 82 | 202 |

**Correlación Pearson frecuencia_gold ↔ F1 (n = 5) = +0.587.** La rareza explica *parte* del F1,
pero no todo: MISOGYNY-NSV no es la categoría más rara y sin embargo tiene el peor F1.

### 3.2 Co-ocurrencia en el gold (nº de memes con ambas categorías)

|  | IDE | STE | OBJ | SXV | MIS |
|---|--:|--:|--:|--:|--:|
| **IDE** | 239 | 202 | 137 | 58 | 113 |
| **STE** | 202 | 301 | 193 | 98 | 144 |
| **OBJ** | 137 | 193 | 262 | 131 | 124 |
| **SXV** | 58 | 98 | 131 | 153 | 77 |
| **MIS** | 113 | 144 | 124 | 77 | 186 |

### 3.3 P(categoría predicha | categoría gold) — la off-diagonal revela confusión

| gold ↓ / pred → | IDE | STE | OBJ | SXV | MIS |
|---|--:|--:|--:|--:|--:|
| **IDE** | 0.77 | 0.73 | 0.44 | 0.29 | 0.59 |
| **STE** | 0.68 | 0.73 | 0.56 | 0.40 | 0.58 |
| **OBJ** | 0.55 | 0.65 | 0.72 | 0.58 | 0.50 |
| **SXV** | 0.37 | 0.51 | 0.75 | 0.77 | 0.43 |
| **MIS** | 0.66 | 0.71 | 0.62 | 0.45 | 0.65 |

**Diagnóstico.** MISOGYNY-NSV es el *outlier*: peor F1 sin ser la más rara ⇒ **confusión de
frontera**. Cuando el gold es MISOGYNY-NSV, el modelo predice también STEREOTYPING (0.71) e
IDEOLOGICAL (0.66). Otras confusiones fuertes: SEXUAL-VIOLENCE ↔ OBJECTIFICATION (0.75) e
IDEOLOGICAL ↔ STEREOTYPING (0.73), coherentes con la altísima co-ocurrencia real (IDEOLOGICAL y
STEREOTYPING aparecen juntas en 202 memes). Figura: `figuras/confusion_2_3.png`.

---

## 4. Ablación del aporte de Gemini (Tarea 4)

Mismo checkpoint, sin reentrenar. En **2.2** se ponen a cero las 7 *features* numéricas de Gemini
en el *forward*. En **2.3** el modelo **no consume ninguna *feature* numérica de Gemini** (solo
texto 768 + EEG 256 + Ekman 7); la ablación equivalente es retirar el texto de Gemini y dejar
solo el OCR.

| Condición | F1-macro | ICM | ICM-Norm | ICM-Soft | F1 [NO/DIR/JUD] |
|---|--:|--:|--:|--:|---|
| 2.2 normal (con 7 gfeat de Gemini) | 0.5585 | +0.0086 | 0.503 | −0.6787 | 0.78 / 0.64 / 0.26 |
| 2.2 ablación (gfeat = 0) | 0.5398 | −0.0243 | 0.491 | −0.7963 | 0.78 / 0.62 / 0.22 |
| 2.3 normal (texto con Gemini) | 0.6741 | +0.2623 | 0.556 | −2.7553 | — |
| **2.3 ablación (solo OCR, sin texto Gemini)** | **0.5151** | **−1.2259** | 0.239 | −6.1976 | — |

**Lectura.** En **2.2**, las 7 *features* numéricas (`sexist_prob, confidence, P_NO, P_DIRECT,
P_JUDG, irony_flag, irony_conf`) aportan poco pero de forma positiva: al eliminarlas, F1-macro
baja −0.019 e ICM −0.033 (el efecto se concentra en JUDGEMENTAL, 0.26 → 0.22). En **2.3**, el
texto de Gemini es el **motor del modelo**: al dejar solo el OCR, F1-macro cae 0.674 → 0.515 e
ICM se desploma +0.262 → −1.226. Esto justifica cuantitativamente el uso de Gemini como
enriquecimiento textual en el pipeline.

> **Aclaración para revisores.** El modelo 2.3 no admite "poner 6 *features* a cero" porque esas
> entradas numéricas no existen; las 11 *disagreement features* son la *salida* de una tarea
> auxiliar, no una entrada del modelo.

---

## 5. Tamaño de los modelos (Tarea 5)

Parámetros totales por checkpoint (backbone incluido), contados desde el `state_dict` guardado;
se excluyen buffers no-parámetro para igualar `sum(p.numel() for p in model.parameters())`.

| Checkpoint | dir | Total (M) | Backbone texto (M) | Cabezas+ (M) | Fichero (MB) |
|---|---|--:|--:|--:|--:|
| M3_vista_E_best (2.1 zip) | checkpoints | 278.659 | 278.044 | 0.616 | 558.6 |
| vista_e_task22_best (2.2 zip) | checkpoints | 278.663 | 278.044 | 0.620 | 558.6 |
| vista_e_task23_best (2.3 zip) | checkpoints | 278.428 | 278.044 | 0.385 | 557.7 |
| vista_e_task21_max512 | _alt | 278.659 | 278.044 | 0.616 | 558.6 |
| vista_e_task21_max512_R | _alt | 278.659 | 278.044 | 0.616 | 558.6 |
| vista_e_task21_longformer | _alt | 281.412 | 280.796 | 0.616 | 564.1 |
| vista_e_task21_longformer_R | _alt | 281.412 | 280.796 | 0.616 | 564.1 |
| vista_e_task22_max512 | _alt | 278.663 | 278.044 | 0.620 | 558.6 |
| vista_e_task22_max512_R | _alt | 278.663 | 278.044 | 0.620 | 558.6 |
| vista_e_task22_longformer | _alt | 281.416 | 280.796 | 0.620 | 564.1 |
| vista_e_task23_max512 | _alt | 278.795 | 278.044 | 0.751 | 559.2 |
| vista_e_task23_max512_v2 | _alt | 278.795 | 278.044 | 0.751 | 559.2 |
| vista_e_task23_max512_R | _alt | 278.795 | 278.044 | 0.751 | 559.2 |
| vista_e_task23_longformer | _alt | 281.547 | 280.796 | 0.751 | 564.7 |
| vista_e_task23_longformer_v2 | _alt | 281.547 | 280.796 | 0.751 | 564.7 |
| vista_e_task23_longformer_R | _alt | 281.547 | 280.796 | 0.751 | 564.7 |

**Lectura.** Familias XLM-R base ≈ 278.4–278.8 M; familias Longformer ≈ 281.4–281.5 M. El coste
lo domina el backbone de texto; las cabezas (jerárquica en 2.2, multietiqueta + auxiliar en 2.3)
añaden menos de 0.8 M. Rango total: 278 428 166 – 281 547 271 parámetros.

---

## 6. Conclusiones

**Rendimiento.**

- **2.1** está resuelto: F1⁺ homogéneo 0.86–0.88; el ensemble con Gemini maximiza el ICM-Soft.
  No hay ganancia decisiva por mayor contexto ni por *reasoning*.
- **2.2** es la subtarea más difícil (ICM ≈ 0). El *threshold* ponderado bate al `argmax` y el
  *blend* con Gemini ayuda en soft; la variante Longformer es la única con ICM claramente
  positivo. JUDGEMENTAL sigue siendo el cuello de botella.
- **2.3** alcanza F1-macro 0.70–0.71 en las mejores variantes, muy por encima de Gemini crudo.

**Calibración, errores y aporte de Gemini.**

- Los modelos están razonablemente calibrados en crudo (ECE 0.09–0.13); **Platt por clase mejora
  el ECE ~70 %** en 2.2/2.3.
- El error dominante en 2.3 es **confusión de frontera** entre categorías semánticamente
  próximas, no la escasez de datos (corr. freq-F1 = +0.59).
- **Gemini** aporta poco como *features* numéricas (2.2) pero es **determinante como texto en
  2.3**: es el argumento cuantitativo del diseño del pipeline.
- **Tamaño:** ~278–281 M parámetros, dominado por el backbone.

---

## 7. Limitaciones (declaradas explícitamente)

1. **Todas las métricas son sobre VALIDACIÓN (n = 598), nunca sobre TEST.** El test de EXIST no
   tiene gold público; cualquier métrica sobre test sería imposible sin inventar. Se generan las
   predicciones de test (submissions / *zip*) pero no se evalúan.
2. **Calibración *in-sample*:** Platt y *temperature* se ajustan sobre la misma validación donde
   se mide el ECE (no hay un tercer conjunto con gold). Las mejoras de ECE son cotas superiores
   optimistas.
3. **Definición de gold hard:** voto mayoritario a 0.5; en 2.3 una categoría se considera
   presente si la marca más de 1/6 de los anotadores (coherente con el código original). Otras
   binarizaciones darían cifras algo distintas.
4. **El modelo 2.3 principal no tiene cabeza de "sexista"**: su ICM se calculó con una compuerta
   = máxima probabilidad de categoría; su ICM-Soft no es 1:1 comparable con las variantes de 6
   salidas.
5. **La ablación de 2.3 es una ablación de TEXTO** (no de *features* numéricas, que no existen en
   ese modelo).
6. Se corrigieron dos incidencias del *harness* de evaluación (nombre `load_t23` vs `load_task23`
   y tokenizers Longformer que requerían `sentencepiece` + `protobuf`) **sin reentrenar ni editar
   los scripts originales**.

---

## Anexo — Ficheros de resultados

Tablas: `metricas_18_modelos.csv`, `calibracion.csv`, `errores_2_3.csv`, `cooc_gold_2_3.csv`,
`cooc_pred_2_3.csv`, `confus_goldcat_predcat_2_3.csv`, `ablacion_gemini_features.csv`,
`parametros.csv`, `task21_variants.csv`, `task22_variants.csv`.
Figuras: `figuras/reliability_2_1.png`, `figuras/reliability_2_2_perclass.png`,
`figuras/reliability_2_3_percat.png`, `figuras/confusion_2_3.png`.
Logs crudos de cada ejecución en `logs/`.

*EXIST 2026 · Task 2 · Equipo Ordantis · Análisis de revisión sobre validación (n = 598) · Sin
reentrenamiento · Ninguna cifra estimada.*
