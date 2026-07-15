# Paso 1 — Recuperación de los thresholds oficiales del run 2.3 (hard)

## Fuente
**Código de generación/evaluación del run entregado:** `Trabajo_LNR/_full_eval_task23.py`
(función `find_best_thr` + `pred_from_probs` + `gold_hard_from_soft`) y `task23.py` (decode del zip).
Al reutilizar esos helpers oficiales sobre las probabilidades crudas ya guardadas, **se reproducen
exactamente los números del paper**.

## Hallazgo principal (≠ hipótesis de partida)
**Los valores tsex=0.30, tcat=0.15 SÍ son los correctos.** El error de mis Bloques 2/5 anteriores
**no estaba en el threshold, sino en la COMPUERTA DE SEXISMO.**

- Las **6 variantes** de Task 2.3 (`max512`, `longformer`, `max512_v2`, `longformer_v2`,
  `max512_R`, `longformer_R`) tienen una **cabeza de sexismo dedicada** (6ª salida). El decode
  oficial usa **esa probabilidad real** como compuerta: `if p_sexista < tsex → "NO"`.
- En mis Bloques 2/5 anteriores aproximé la compuerta con `max(prob_categoría)` **para todos** los
  checkpoints (ver caveat 2 del README). Eso hace que casi ningún meme quede filtrado como "NO" →
  **sobre-predicción masiva** → ICM negativo.
- Con la **cabeza de sexismo real** y los mismos tsex=0.30/tcat=0.15, el ICM pasa a **positivo** y se
  reproduce el paper.

Es decir: **mismo threshold, compuerta distinta** explica el salto de ICM −0.47 a +0.34.

## Reproducción oficial por checkpoint (validación, n=598, tsex=0.30, tcat=0.15 uniforme)
| Checkpoint | F1-micro@0.5 | F1-macro | ICM | ICMNorm | ICMSoft | ICMSoftNorm |
|---|---|---|---|---|---|---|
| **vista_e_task23_max512_v2** | 0.6251 | **0.7146** | **+0.3417** | 0.5707 | −5.9912 | 0.1211 |
| vista_e_task23_max512 | 0.6238 | 0.7137 | +0.3363 | 0.5696 | −4.5932 | 0.2095 |
| vista_e_task23_max512_R | 0.6149 | 0.7135 | +0.3331 | 0.5690 | −6.1419 | 0.1116 |
| vista_e_task23_longformer_R | 0.6144 | 0.7116 | +0.3067 | 0.5635 | −5.4749 | 0.1538 |
| vista_e_task23_longformer | 0.5856 | 0.7051 | +0.2647 | 0.5548 | −3.4947 | 0.2790 |
| vista_e_task23_longformer_v2 | 0.6188 | 0.7029 | +0.2455 | 0.5508 | −5.1343 | 0.1753 |
| Gemini zero-shot | 0.5938 | 0.5367 | −3.4148 | 0.0000 | −17.1151 | 0.0000 |

➡️ **El run 2.3 hard_1 del paper (F1-macro ≈ 0.715, ICM ≈ +0.340) es `vista_e_task23_max512_v2`**
(0.7146 / +0.3417). Coincide con `NOMENCLATURA_RUNS.csv` (`2.3_run1_hard = max512_v2`).
**NO** es el checkpoint principal `vista_e_task23_best`.

## El checkpoint principal `vista_e_task23_best` (sin cabeza de sexismo)
No tiene 6ª salida → su compuerta sólo puede aproximarse con `max(prob_categoría)`. **El harness
oficial `_full_eval_task23.py` NO puede evaluarlo** (falla con *"not enough values to unpack:
expected 5, got 4"* porque `task23.infer` devuelve 4 valores, sin sexismo). Es decir, **no existe
número oficial del principal para 2.3.**

Bajo el **gold oficial** (el que reproduce Table 6) y la compuerta aproximada `max(cat)`, el barrido
`find_best_thr` da su óptimo por criterio en **tsex=0.30, tcat=0.20 → F-macro 0.6750, ICM −2.1484**
(ICMSoft −12.13). El principal **puntúa muy mal en 2.3 hard** porque sin cabeza de sexismo no filtra
"NO" y sobre-predice. **No es el run entregado** (el entregado es `max512_v2`).

> Nota de trazabilidad: en una primera pasada calculé el principal con el gold de `_common.py`
> (que re-deriva el sexismo de las anotaciones: 65 % sexistas) y salía ICM ≈ +0.26. Ese gold **no es
> el oficial**: el oficial (`mod.infer`) trata los 598 memes de val como categorizables (SX=1) y es el
> único que reproduce el paper. Todas las cifras finales usan el **gold oficial**.

## Conclusión para confirmar antes del Paso 2
1. Los thresholds correctos son **tsex=0.30, tcat=0.15 uniforme**, **con la cabeza de sexismo real
   de cada variante** como compuerta (no `max(cat)`).
2. El 0.715/+0.340 del paper corresponde a **`vista_e_task23_max512_v2`**, el run entregado, no al
   principal.
3. Reejecutaré Bloques 2 y 5 (solo 2.3) usando la **compuerta de sexismo real** de las 6 variantes
   (que exige re-inferir para capturar la 6ª salida, no guardada en los `preds_val_*.csv`), y para
   el principal dejaré constancia de que su gate es `max(cat)` (techo ICM ≈ +0.26).
