# Resumen ejecutivo — Análisis (validación, n=598)

*Nomenclatura: modelos por su run entregado; variantes con etiqueta descriptiva de config. Ver `TERMINOLOGIA_PAPER.md`.*

## Los 3 mejores configs por subtarea y métrica

**Tarea 2.1 (sexista sí/no)** — homogénea y bien resuelta:
| Métrica | 1º | 2º | 3º |
|---|---|---|---|
| F1⁺ | Longformer +reason (0.879) | XLM-R/512 (0.872) | Longformer (0.868) |
| ICM | XLM-R/512 +reason (+0.409) | Longformer (+0.401) | XLM-R/512 (+0.394) |
| ICM-Soft | Longformer (+0.543) | Longformer +reason (+0.533) | XLM-R/512 +reason (+0.388) |
> *El ensemble 0.6·Vista E base + 0.4·Gemini alcanza el mejor ICM-Soft global (+0.596).*

**Tarea 2.2 (intención)** — la subtarea difícil, dominada por una config:
| Métrica | 1º | 2º | 3º |
|---|---|---|---|
| F1-macro | **Longformer (0.607)** | XLM-R/512 +reason (0.567) | Vista E base (0.559) |
| ICM | **Longformer (+0.113)** | Vista E base (+0.009) | XLM-R/512 +reason (−0.000) |
> *El Longformer es el único con ICM claramente positivo; el resto ronda o baja de 0.*

**Tarea 2.3 (categorización, decode oficial: cabeza de sexismo real + `find_best_thr`)** — reproduce Table 6:
| Métrica | 1º | 2º | 3º |
|---|---|---|---|
| F1-macro | **XLM-R/512 +bal (0.7146)** | XLM-R/512 (0.7137) | XLM-R/512 +bal +reason (0.7135) |
| ICM | **XLM-R/512 +bal (+0.3417)** | XLM-R/512 (+0.3363) | XLM-R/512 +bal +reason (+0.3331) |
> *La familia **XLM-R/512** domina 2.3 (su cabeza de sexismo filtra mejor los "NO"). **XLM-R/512 +bal**
> es el run entregado **2.3 hard_1** (paper: 0.715 / +0.340). La Vista E base (320), sin cabeza de
> sexismo, no es comparable (ICM −2.15 con gate `max(cat)`).*

## Efecto de la calibración (Platt por clase, macro OvR)
| Subtarea | ECE crudo | ECE Platt | Reducción |
|---|---|---|---|
| 2.2 (Vista E base) | 0.1314 | 0.0367 | **−72 %** |
| 2.3 (**2.3 hard_1 = XLM-R/512 +bal**) | 0.0784 | 0.0333 | **−58 %** |

En 2.1 el modelo ya está bien calibrado (ECE ≈ 0.09–0.12); el *temperature scaling* **empeora** el ECE-hard y solo el *blend* con Gemini lo mejora (0.108 → 0.075).

> *Nota: la binarización del gold depende de la subtarea — 2.1 a 0.5, 2.2 por mayoría, y **2.3 a 1/6**,
> el mismo umbral que la evaluación oficial (Bloques 2/5). Los ECE de 2.3 son, por tanto, consistentes
> con esa evaluación (verificado byte-idéntico al re-ejecutar).*

## Ablación de features numéricas de Gemini (Δ = con − sin, decode oficial)
| Subtarea | Δ ICM al quitar | Interpretación |
|---|---|---|
| 2.2 (4 configs) | **−0.03 a −0.12** | Las 7 features numéricas aportan de forma consistente y notable. |
| 2.3 (6 configs) | **−0.002 a −0.037** | Las 6 features numéricas aportan **poco pero consistente** (quitarlas cuesta ICM en las 6). Mayor efecto en Longformer. |

## Resultados inesperados (lo más relevante para el paper)
1. **La cabeza de sexismo es decisiva en 2.3:** con la compuerta de sexismo REAL (6ª salida) las 6
   variantes dan ICM **positivo** (+0.25 a +0.34) y F1-macro ~0.70–0.71. Sin ella (aproximación
   `max(cat)` del principal), el modelo sobre-predice y el ICM se hunde (−2.15). El filtrado de "NO"
   es el factor crítico, no el umbral de categoría.
2. **La fisiología (EEG+Ekman) PERJUDICA a 2.2 y a 2.3:** al ponerla a cero, en 2.2 F1-macro sube 0.559→0.576 e ICM +0.009→+0.077; en 2.3 F1-macro sube 0.675→0.688 e ICM −2.15→−2.00. En 2.1 es irrelevante. Conclusión limpia: **el EEG+Ekman no ayuda en ninguna subtarea.**
3. **Las features numéricas de Gemini aportan poco pero consistente en 2.3** (quitarlas cuesta ICM en las 6 configs): el grueso del aporte de Gemini a 2.3 es *textual*, con un extra numérico pequeño y real.
4. **El *temperature scaling* no ayuda en 2.1**: el modelo ya está calibrado (T≈1.1–1.3 empuja hacia 0.5 y empeora el ECE-hard).
5. **MISOGYNY-NON-SEXUAL-VIOLENCE sigue siendo la categoría más débil** de 2.3 (en el run 2.3 hard_1 = XLM-R/512 +bal: F1 0.62, precisión 0.50, la más baja), por rareza y confusión de frontera con STEREOTYPING e IDEOLOGICAL — pero ya no catastrófica una vez corregida la compuerta.
