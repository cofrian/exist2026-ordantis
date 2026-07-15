# TAREA 3 - Analisis de errores 2.3 (modelo principal, validacion n=598)

Umbrales usados (optimizados sobre val en eval_main23): thr_sex=0.34, thr_cat=0.20

## F1 por categoria (ordenado por frecuencia gold ascendente)

| categoria | freq_gold | F1 | precision | recall | FP | FN |
|---|---|---|---|---|---|---|
| SEXUAL-VIOLENCE | 153 | 0.676 | 0.602 | 0.771 | 78 | 35 |
| MISOGYNY-NON-SEXUAL-VIOLENCE | 186 | 0.556 | 0.488 | 0.645 | 126 | 66 |
| IDEOLOGICAL-INEQUALITY | 239 | 0.688 | 0.622 | 0.770 | 112 | 55 |
| OBJECTIFICATION | 262 | 0.712 | 0.707 | 0.718 | 78 | 74 |
| STEREOTYPING-DOMINANCE | 301 | 0.712 | 0.698 | 0.728 | 95 | 82 |

Correlacion frecuencia_gold vs F1 (Pearson, n=5 cats): r = +0.587.
Si r es alto y positivo -> el F1 bajo se explica por RAREZA (pocas muestras).
Las off-diagonales de 'P(pred cat | gold cat)' revelan CONFUSION DE FRONTERA
(el modelo predice otra categoria cuando la gold es X).

## Confusiones de frontera mas fuertes (off-diagonal de P(pred|gold))
- gold=SEXUAL-VIOLENCE -> predice tambien OBJECTIFICATION: 0.75
- gold=IDEOLOGICAL-INEQUALITY -> predice tambien STEREOTYPING-DOMINANCE: 0.73
- gold=MISOGYNY-NON-SEXUAL-VIOLENCE -> predice tambien STEREOTYPING-DOMINANCE: 0.71
- gold=STEREOTYPING-DOMINANCE -> predice tambien IDEOLOGICAL-INEQUALITY: 0.68
- gold=MISOGYNY-NON-SEXUAL-VIOLENCE -> predice tambien IDEOLOGICAL-INEQUALITY: 0.66
- gold=OBJECTIFICATION -> predice tambien STEREOTYPING-DOMINANCE: 0.65