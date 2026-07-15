# Terminología del paper — nombres fijados (antes de escribir)

Decisión (2026-07-04):
- **Modelos entregados → se nombran por su RUN de submission** (`2.1 run1 hard`, `2.3 hard_1`, …).
- **Variantes de ablación → etiquetas DESCRIPTIVAS por config** (no sufijos técnicos).

## 1. Diccionario de config (piezas atómicas)
Las etiquetas se construyen combinando **backbone + tokens** con los modificadores que aplique:
- **`XLM-R/512`** = XLM-R base, max_len 512. **`Longformer`** = XLM-R-Longformer 4096.
- **`+reason`** = sufijo `_R`: se añade el texto de *reasoning* de Gemini al input.
- **`+bal`** = sufijo `_v2`: muestreo balanceado (solo existe en 2.3).
- **`Vista E base`** = checkpoint principal warm-start (`M3_vista_E_best` / `*_best`, max_len 256–320).

**Ojo, el sufijo `_R` combina distinto por subtarea:**
- 2.1 / 2.2: `_R` = **+reason** (p.ej. `max512_R` → *XLM-R/512 +reason*).
- 2.3: los runs `_v2` = **+bal** y `_R` = **+bal +reason** (el `_R` de 2.3 lleva también el sampler).

## 2. Runs entregados → checkpoint → etiqueta descriptiva
**El "modelo principal" reportado por subtarea es el RUN entregado, no el checkpoint `*_best`.**
En 2.3, el número de Table 6 (F1-macro 0.7146 / ICM +0.3417) es **`2.3 hard_1`** = `max512_v2` =
**XLM-R/512 +bal**. El checkpoint `vista_e_task23_best` (Vista E base 320) **no es un run entregado en
2.3** y no debe reportarse como el modelo de 2.3.

### Tarea 2.1 (sexista sí/no)
| Run | Checkpoint | Etiqueta descriptiva | F1⁺ | ICM |
|---|---|---|---|---|
| 2.1 hard_1 | vista_e_task21_max512_R | XLM-R/512 +reason | 0.863 | +0.409 |
| 2.1 hard_2 | vista_e_task21_longformer | Longformer | 0.868 | +0.401 |
| 2.1 hard_3 | vista_e_task21_max512 | XLM-R/512 | 0.872 | +0.394 |
| 2.1 soft_1 | M3_vista_E_best | Vista E base + blend Gemini | 0.856 | +0.373 |
| 2.1 soft_2 | vista_e_task21_longformer | Longformer (directo) | 0.868 | +0.401 |
| 2.1 soft_3 | vista_e_task21_longformer_R | Longformer +reason | 0.879 | +0.380 |

### Tarea 2.2 (intención)
| Run | Checkpoint | Etiqueta descriptiva | F1-macro | ICM |
|---|---|---|---|---|
| 2.2 hard_1 | vista_e_task22_longformer | Longformer | 0.605 | +0.098 |
| 2.2 hard_2 | vista_e_task22_max512 | XLM-R/512 | 0.550 | +0.079 |
| 2.2 hard_3 | vista_e_task22_best + blend Gemini | Vista E base + blend | 0.584 | +0.023 |
| 2.2 soft_1 | vista_e_task22_best + blend Gemini | Vista E base + blend | — | ICMSoft −0.157 |
| 2.2 soft_2 | vista_e_task22_longformer + Platt | Longformer + Platt | — | ICMSoft −0.329 |
| 2.2 soft_3 | vista_e_task22_max512 + Platt | XLM-R/512 + Platt | — | ICMSoft −0.439 |

### Tarea 2.3 (categorización) — decode oficial, gate sexismo real
| Run | Checkpoint | Etiqueta descriptiva | F1-macro | ICM |
|---|---|---|---|---|
| **2.3 hard_1** | vista_e_task23_max512_v2 | **XLM-R/512 +bal** | **0.7146** | **+0.3417** |
| 2.3 hard_2 | vista_e_task23_max512 | XLM-R/512 | 0.7137 | +0.3363 |
| 2.3 hard_3 | vista_e_task23_max512_R | XLM-R/512 +bal +reason | 0.7135 | +0.3331 |
| 2.3 soft_1 | vista_e_task23_longformer | Longformer | 0.7051 | +0.2647 |
| 2.3 soft_2 | vista_e_task23_max512 | XLM-R/512 | 0.7137 | +0.3363 |
| 2.3 soft_3 | vista_e_task23_longformer_v2 | Longformer +bal | 0.7029 | +0.2455 |

## 3. Regla de estilo para el texto
- Al citar un resultado de submission: **"run 2.3 hard_1 (XLM-R/512 +bal)"**.
- En tablas de ablación: solo la **etiqueta descriptiva** (XLM-R/512, Longformer +bal, …).
- Reservar "Vista E" para la **arquitectura** (texto+EEG+Ekman); no usarlo como nombre de un run concreto.
- Nunca usar en el paper los identificadores internos (`vista_e_task23_max512_v2`); van solo en el
  anexo de reproducibilidad / código.
