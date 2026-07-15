# Informe analítico completo — 16 checkpoints EXIST 2026 Task 2 (Ordantis)

> Análisis exhaustivo sobre **validación (n=598)**, sin reentrenar. Todas las cifras salen de ejecutar
> código sobre datos reales (predicciones del Bloque 0 + PyEvALL 0.2.11). Ver `README_resultados.md`
> para trazabilidad y `resumen_ejecutivo.md` para la versión de una página.

---

## Bloque 0 — Predicciones base (16 checkpoints)

Se generaron las probabilidades crudas de los 598 memes de validación para los 16 checkpoints
(`preds_val_*.csv`). Verificación: 598 filas por fichero, mismos IDs, todo en [0,1], sin NaN ni
duplicados. Las medias son coherentes con la naturaleza de cada tarea: 2.1 `P_YES`≈0.56–0.63;
2.2 media 0.333 exacta (las 3 clases suman 1 → cabeza jerárquica correcta); 2.3 ≈0.24–0.29 por
categoría (multi-etiqueta).

**Conclusión.** La base de predicciones es sólida y reproducible; todo el análisis posterior se
apoya en ella sin volver a inferir (salvo Bloques 3 y 6, que requieren forward modificado).

---

## Bloque 4 — Tamaño y coste

| Familia | Nº ckpts | Params totales | Backbone | Cabezas |
|---|---|---|---|---|
| XLM-R base | 10 | 278.4–278.8 M | 278.04 M | 0.38–0.75 M |
| Longformer | 6 | 281.4–281.5 M | 280.80 M | 0.62–0.75 M |

**Conclusión.** El coste está **totalmente dominado por el backbone de texto** (>99.7 % de los
parámetros). Las cabezas (jerárquica en 2.2, multi-etiqueta + auxiliar en 2.3) añaden <0.8 M. La
diferencia entre familias (~3 M) es el mayor contexto del Longformer. **Implicación:** elegir
Longformer vs XLM-R es una decisión de *contexto*, no de *capacidad* — el número de parámetros
entrenables efectivos es casi idéntico. El Longformer gana en 2.2 (contexto largo); en 2.3 gana la
familia XLM-R/512 (ver Bloque 5), así que su sobrecoste (~1 %) solo está justificado para 2.2.

---

## Bloque 5 — Rendimiento oficial por checkpoint (PyEvALL, validación)

### 2.1 — sexista sí/no
| Config (run) | F1⁺ | AUC | ICM | ICM-Soft |
|---|---|---|---|---|
| Vista E base (2.1 soft_1, base del blend) | 0.8609 | 0.8839 | +0.3855 | +0.4798 |
| XLM-R/512 (2.1 hard_3) | 0.8715 | 0.8852 | +0.3944 | +0.3025 |
| XLM-R/512 +reason (2.1 hard_1) | 0.8633 | 0.8925 | **+0.4088** | +0.3878 |
| Longformer (2.1 hard_2 / soft_2) | 0.8676 | 0.8833 | +0.4008 | **+0.5428** |
| Longformer +reason (2.1 soft_3) | **0.8790** | 0.8843 | +0.3801 | +0.5334 |
| *blend 0.6·Vista E + 0.4·Gemini* | 0.8560 | 0.8886 | +0.3733 | **+0.5957** |
| *Gemini zero-shot* | 0.8748 | 0.8550 | +0.3429 | +0.2335 |

**Conclusión.** 2.1 está **resuelta y es robusta**: F1⁺ 0.86–0.88 y AUC ~0.88–0.89 muy homogéneos.
Ningún config domina en todo: *XLM-R/512 +reason* gana en ICM, *Longformer +reason* en F1⁺, *Longformer*
en ICM-Soft. Gemini zero-shot compite en F1⁺ (0.875) pero es el peor en calidad probabilística
(ICM-Soft +0.234), lo que confirma que **sus probabilidades están mal calibradas** (ver Bloque 1).
El *blend* con Gemini es la mejor decisión soft (ICM-Soft +0.596).

### 2.2 — intención (NO/DIRECT/JUDGEMENTAL)
| Config (run) | F1-macro (thr) | ICM | ICM-Soft |
|---|---|---|---|
| Vista E base (2.2 hard_3 / soft_1, base del blend) | 0.5585 | +0.0086 | −0.6788 |
| XLM-R/512 (2.2 hard_2) | 0.5456 | −0.0645 | −0.6858 |
| XLM-R/512 +reason | 0.5669 | −0.0002 | −0.8021 |
| **Longformer (2.2 hard_1 / soft_2)** | **0.6073** | **+0.1129** | −0.6758 |
| *blend con Gemini* | 0.5842 | +0.0217 | −0.1572 |
| *Gemini zero-shot* | 0.5541 | −0.0711 | −2.5618 |

**Conclusión.** 2.2 es la **subtarea más difícil**: el ICM ronda 0 (el modelo apenas supera el
baseline jerárquico de PyEvALL) y el ICM-Soft es fuertemente negativo. El **Longformer es el claro
ganador** (ICM +0.113, único positivo real), lo que sugiere que **la intención necesita contexto
largo** (el texto enriquecido de Gemini con *reasoning* llega a ~1000 tokens). El *blend* con
Gemini es lo único que rescata el ICM-Soft (−0.157 vs −0.68 crudo).

### 2.3 — categorización (decode oficial: cabeza de sexismo real + `find_best_thr`, reproduce Table 6)
*(nombres de run entregado + etiqueta descriptiva; ver `TERMINOLOGIA_PAPER.md`)*
| Run / config | F1-macro | F1-micro | ICM | ICM-Soft | thr |
|---|---|---|---|---|---|
| **2.3 hard_1 · XLM-R/512 +bal** | **0.7146** | 0.7234 | **+0.3417** | −5.9912 | tsex0.30/tcat0.15 |
| 2.3 hard_2 · XLM-R/512 | 0.7137 | 0.7224 | +0.3363 | −4.5932 | tsex0.30/tcat0.15 |
| 2.3 hard_3 · XLM-R/512 +bal +reason | 0.7135 | 0.7205 | +0.3331 | −6.1419 | tsex0.30/tcat0.15 |
| Longformer +bal +reason | 0.7116 | 0.7243 | +0.3067 | −5.4749 | tsex0.30/tcat0.15 |
| 2.3 soft_1 · Longformer | 0.7051 | 0.7179 | +0.2647 | −3.4947 | tsex0.30/tcat0.15 |
| 2.3 soft_3 · Longformer +bal | 0.7029 | 0.7140 | +0.2455 | −5.1343 | tsex0.30/tcat0.15 |
| *Vista E base (320), sin cabeza sexismo* | 0.6750 | 0.6821 | −2.1484 | −12.1338 | tsex0.30/tcat0.20 (gate max-cat) |
| *Gemini zero-shot* | 0.5974 | 0.6056 | −0.5943 | −8.2406 | — |

**Conclusión.** Con la **compuerta de sexismo real** de cada variante, 2.3 pasa a tener **ICM positivo**
(+0.25 a +0.34) y F1-macro ~0.70–0.71, **reproduciendo Table 6** del paper (el **run 2.3 hard_1 =
XLM-R/512 +bal** = 0.7146 / +0.3417). Aquí **la familia XLM-R/512 supera a Longformer** (al revés
que en 2.2): en 2.3 lo decisivo no es el contexto largo sino que la **cabeza de sexismo filtre bien
los "NO"**. La *Vista E base (320)* **carece de esa cabeza** → con la aproximación `max(cat)`
sobre-predice y se hunde (ICM −2.15); no es 1:1 comparable ni es un run entregado en 2.3. El ICM-Soft sigue
muy negativo en todos (la distribución soft de categorías es difícil de calibrar), y Gemini zero-shot
es el peor en soft (−8.2).

---

## Bloque 1 — Calibración

### Resumen (ECE-hard, menor = mejor)
| Subtarea | crudo (principal) | calibrado | mejor variante cruda |
|---|---|---|---|
| 2.1 | 0.1084 | temp **0.133** (empeora) / blend **0.075** (mejora) | Longformer 0.093 |
| 2.2 (macro OvR, Vista E base) | 0.1314 | Platt **0.0367** (−72 %) | — |
| 2.3 (macro OvR, **2.3 hard_1 = XLM-R/512 +bal**) | 0.0784 | Platt **0.0333** (−58 %) | Longformer +bal 0.058 |

> **Nota (binarización del gold para calibración):** el umbral de binarización depende de la subtarea:
> **2.1** usa 0.5 (mayoría YES), **2.2** usa la clase mayoritaria (argmax), y **2.3 usa 1/6 por
> categoría — el mismo umbral que la evaluación oficial** de Bloques 2/5. Como además el soft de
> categorías es idéntico entre golds (Δ=0) y el gold oficial trata los 598 memes como sexistas (sin
> compuerta), **los ECE de 2.3 son consistentes con la evaluación oficial** (verificado: re-ejecutar el
> bloque produce un CSV byte-idéntico). Las mejoras por Platt/temperature son in-sample (cotas
> superiores optimistas).

**Conclusiones:**
1. **Los modelos ya están razonablemente calibrados en crudo** (ECE 0.06–0.15). No hay
   sobreconfianza masiva.
2. **Platt por clase es muy efectivo en 2.2 y 2.3** (reduce el ECE −72 % en 2.2 y −58 % en 2.3), a
   costa de a veces empeorar el soft-ECE (se ajusta a etiquetas duras). Peor clase cruda: JUDGEMENTAL
   en 2.2. **La calibración de 2.3 se reporta sobre el run 2.3 hard_1 (XLM-R/512 +bal)** (el entregado
   que reproduce Table 6), no sobre la Vista E base (320).
3. **El *temperature scaling* NO ayuda en 2.1**: T estimada 1.1–1.3 (>1 empuja las probabilidades
   hacia 0.5), lo que **sube** el ECE-hard aunque mejore el soft. Señal de que el modelo ya está
   bien calibrado para decisión dura.
4. **Gemini es el peor calibrado** en las tres subtareas (2.1 MCE 0.55; 2.2 ECE macro 0.171; 2.3
   0.107) — sus probabilidades son extremas (cercanas a 0/1) y no reflejan la incertidumbre real.
5. **Caveat honesto:** Platt/temperature se ajustan in-sample; las mejoras son cotas superiores.

---

## Bloque 2 — Análisis de errores 2.3 (7 configs, decode oficial + gate sexismo real)

### Global por config (mejores)
| Config (run) | F1-macro | F1-micro | ICM | ICM-Soft |
|---|---|---|---|---|
| **XLM-R/512 +bal (2.3 hard_1)** | **0.7146** | **0.7234** | **+0.3417** | −5.991 |
| XLM-R/512 (2.3 hard_2) | 0.7137 | 0.7224 | +0.3363 | −4.593 |
| Longformer (2.3 soft_1) | 0.7051 | 0.7179 | +0.2647 | −3.495 |
| *Vista E base (320), gate max-cat* | 0.6750 | 0.6821 | −2.1484 | −12.13 |

### Por categoría del run 2.3 hard_1 (XLM-R/512 +bal) (ordenado por F1)
| Categoría | freq_gold | freq_pred | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| STEREOTYPING-DOMINANCE | 372 | 487 | 0.682 | 0.893 | 0.773 |
| IDEOLOGICAL-INEQUALITY | 346 | 419 | 0.666 | 0.806 | 0.729 |
| OBJECTIFICATION | 324 | 377 | 0.695 | 0.809 | 0.748 |
| SEXUAL-VIOLENCE | 183 | 201 | 0.672 | 0.738 | 0.703 |
| MISOGYNY-NON-SEXUAL-VIOLENCE | 210 | 335 | 0.505 | 0.805 | 0.620 |

**Conclusiones:**
1. **Predicción equilibrada:** con la compuerta real la sobre-predicción desaparece (freq_pred ≈
   freq_gold, no 2–5× como antes). Precisión 0.50–0.70, recall 0.74–0.89, **F1 0.62–0.77 por
   categoría**. El factor que arreglaba el ICM no era subir tcat, sino **filtrar bien los "NO"** con la
   cabeza de sexismo.
2. **MISOGYNY-NSV sigue siendo la más débil** (F1 0.62, precisión 0.50): la más sobre-predicha
   relativa (335 vs 210) y la de peor frontera, pero ya **lejos del colapso** previo.
3. **Confusión de frontera** (matriz P(pred|gold), `confusion_por_checkpoint.csv`): cuando el gold es
   MISOGYNY, el modelo también activa STEREOTYPING (0.89) e IDEOLOGICAL (0.78); IDEOLOGICAL↔
   STEREOTYPING se confunden mutuamente (0.80–0.86). Coherente con la co-ocurrencia real.
4. **Estructura del gold OFICIAL** (`cooc_gold_2_3.csv`, `densidad_multietiqueta_gold.csv`, mismo gold
   que los F1 de arriba → freq por categoría idénticas 346/372/324/183/210):
   - **Co-ocurrencia:** STEREOTYPING es la más frecuente (372) y el eje del solapamiento —
     co-ocurre con IDEOLOGICAL (224) y OBJECTIFICATION (215); OBJECTIFICATION↔SEXUAL-VIOLENCE (141)
     es el otro par fuerte. Coherente con la confusión del modelo.
   - **Densidad multi-etiqueta:** 163 memes con 1 categoría, 162 con 2, 170 con 3, 77 con 4, 26 con 5
     (**media 2.4 categorías/meme**). El gold oficial es marcadamente multi-etiqueta, lo que explica
     que el F1-micro (~0.72) supere al F1-macro y que las categorías raras arrastren el macro.
5. **Gemini es conservador con SEXUAL-VIOLENCE** (`gemini_prob_media_por_categoria.csv`): prob media
   0.092 — infra-estima la categoría más sensible.

---

## Bloque 3 — Ablación de features numéricas de Gemini

### 2.2 (7 features: sexist_prob, confidence, 3×intención, irony_flag, irony_conf)
| Config | F1-macro (con → sin) | ICM (con → sin) |
|---|---|---|
| Vista E base | 0.5585 → 0.5398 (**−0.019**) | +0.0086 → −0.0243 (**−0.033**) |
| XLM-R/512 | 0.5456 → 0.5293 (−0.016) | −0.0645 → −0.0951 (−0.031) |
| XLM-R/512 +reason | 0.5669 → 0.5355 (−0.031) | −0.0002 → −0.1172 (−0.117) |
| Longformer | 0.6073 → 0.5831 (−0.024) | +0.1129 → +0.0702 (−0.043) |

### 2.3 (6 features: sexist_prob + 5 category_probs) — decode oficial (gate real, gold oficial)
*(«con» reproduce Bloque 5 exactamente; Δ = quitar features)*
| Config | F1-macro (con → sin) | ICM (con → sin) |
|---|---|---|
| XLM-R/512 | 0.7137 → 0.7133 (−0.0005) | +0.3363 → +0.3339 (**−0.002**) |
| XLM-R/512 +bal | 0.7146 → 0.7138 (−0.0009) | +0.3417 → +0.3353 (**−0.006**) |
| XLM-R/512 +bal +reason | 0.7135 → 0.7126 (−0.0009) | +0.3331 → +0.3243 (**−0.009**) |
| Longformer | 0.7051 → 0.7024 (−0.0027) | +0.2647 → +0.2399 (**−0.025**) |
| Longformer +bal | 0.7029 → 0.6978 (−0.0051) | +0.2455 → +0.2083 (**−0.037**) |
| Longformer +bal +reason | 0.7116 → 0.7089 (−0.0026) | +0.3067 → +0.2850 (**−0.022**) |

**Conclusiones:**
1. **En 2.2 las features numéricas de Gemini SÍ aportan** de forma consistente: quitarlas baja
   F1-macro (−0.016 a −0.031) e ICM (−0.03 a −0.12) en los 4 checkpoints. Su efecto se concentra en
   la clase difícil JUDGEMENTAL (F1 cae de ~0.27 a ~0.22).
2. **En 2.3 las 6 features numéricas aportan poco pero de forma CONSISTENTE:** con el decode oficial,
   quitarlas cuesta ICM en **las 6 variantes** (−0.002 a −0.037) y baja levemente el F1 (−0.001 a
   −0.005). El efecto es mayor en Longformer (hasta −0.037 ICM) que en XLM-R/512 (~−0.005). *No* son
   irrelevantes —como sugería el decode antiguo— aunque su peso es modesto frente al texto.
3. **Contraste clave para el paper:** el grueso del aporte de Gemini a 2.3 es **textual** (retirar el
   *texto* de Gemini hundía F1-macro 0.674→0.515, hallazgo previo); las *features numéricas* añaden un
   extra **pequeño pero real y consistente**. En 2.2 las features numéricas pesan más (Δ ICM −0.03 a
   −0.12).
4. La *Vista E base (320)* **no consume features numéricas** (solo texto+EEG+Ekman) → la ablación
   numérica es N/A para ella (celda NA, no forzada).

---

## Bloque 6 — Ablación de fisiología (EEG 256 + Ekman 7), modelos base

| Subtarea (config) | Métrica con → sin fisiología | Δ |
|---|---|---|
| 2.1 (Vista E base) | F1⁺ 0.8609 → 0.8613; ICM +0.3855 → +0.3839 | ≈ 0 |
| 2.2 (Vista E base) | F1-macro 0.5585 → **0.5756**; ICM +0.0086 → **+0.0771** | **mejora al quitar** |
| 2.3 (Vista E base 320) | F1-macro 0.6750 → **0.6881**; ICM −2.1484 → **−1.9976** | **mejora al quitar** |

*(2.3 con el decode oficial; «con fisiología» reproduce el principal de Bloque 5: 0.6750 / −2.1484.)*

**Conclusiones:**
1. **En 2.1 la fisiología es irrelevante** (Δ despreciable): la decisión sexista/no se toma con el
   texto; EEG+Ekman no aportan ni restan.
2. **En 2.2 la fisiología PERJUDICA:** al ponerla a cero, F1-macro sube +0.017 y el ICM casi se
   multiplica ×9 (+0.009 → +0.077). Es el **resultado más inesperado**: las señales EEG/Ekman están
   introduciendo ruido en la clasificación de intención. **Recomendación directa para el paper:
   reportar 2.2 sin fisiología, o al menos discutir este efecto.**
3. **En 2.3 la fisiología también PERJUDICA:** con el decode oficial, quitarla sube F1-macro +0.013 e
   ICM +0.151 (−2.148 → −1.998). Coincide en signo con 2.2 (antes, con el decode erróneo, parecía
   ayudar; corregido, estorba).
4. **Lectura transversal:** la fisiología sensorial **no ayuda en ninguna subtarea**: neutra en 2.1 y
   **perjudicial en 2.2 y 2.3**. Es una conclusión limpia y fuerte: la arquitectura común
   "texto+EEG+Ekman" **no está justificada** por los datos; el EEG+Ekman debería retirarse (o
   revisarse) en 2.2 y 2.3.

---

## Conclusiones globales (para el paper)

1. **Cada subtarea premia una cosa distinta:** en **2.2** gana la **longitud de contexto**
   (Longformer, ICM +0.11), mientras que en **2.3** gana la **familia XLM-R/512** (mejor cabeza de
   sexismo, ICM +0.34) por encima de Longformer. 2.1 está resuelta y es homogénea.
2. **La cabeza de sexismo es imprescindible en 2.3:** filtrar bien los "NO" lleva el ICM de −2.15
   (aproximación `max(cat)` del principal) a +0.34 (variantes con 6ª salida). Es el mayor
   determinante del rendimiento en 2.3, por encima del umbral de categoría.
3. **Gemini aporta por dos vías según la subtarea:** en 2.2 pesan sobre todo las *features numéricas*
   (Δ ICM −0.03 a −0.12 al quitarlas); en 2.3 el grueso es el *texto enriquecido*, con un extra
   numérico **pequeño pero consistente** (Δ ICM −0.002 a −0.037 al quitarlas en las 6 configs).
4. **La calibración es buena de base** y mejorable con Platt (2.2/2.3, −65/72 % ECE). Gemini es el
   componente peor calibrado y conviene envolverlo en un *blend* (2.1) más que usarlo crudo en soft.
5. **La fisiología (EEG+Ekman) no ayuda en ninguna subtarea:** neutra en 2.1 y **perjudicial en 2.2
   (ICM +0.009→+0.077) y 2.3 (ICM −2.15→−2.00)** al retirarla. Cuestiona la arquitectura común.
6. **MISOGYNY-NON-SEXUAL-VIOLENCE** es la categoría más débil de 2.3 (F1 0.62 en el run entregado):
   rara y confundida con categorías próximas; margen de mejora vía augmentación o modelado de la
   jerarquía de categorías.

## Limitaciones
- Todo es sobre **validación (n=598)**; el test de EXIST no tiene gold público.
- Calibración **in-sample** (cotas superiores optimistas).
- El modelo 2.3 principal no tiene cabeza de sexismo (compuerta aproximada con max prob).
- Ninguna cifra estimada: todo proviene de ejecutar código sobre datos reales.
