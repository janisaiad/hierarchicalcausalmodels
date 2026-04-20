# STAR Report (Math-Only Alignment)

## Executive summary

Le reporting est aligné sur l'outcome maths `M` (`gktmathss`) pour éviter tout mélange de cadres.

## Baseline benchmark (math)

Fichiers de référence:

- `examples/STAR/results/star_baseline_math_benchmark.json`
- `examples/STAR/star_baseline_math_benchmark.md`

Résultats:

- OLS (effet `small`) entre `+8.10` et `+9.50` points de score maths.
- 2SLS (taille de classe) `-1.2186` point de score maths par élève.

## HCM alignment

- Les analyses HCM doivent être lues contre ces baselines math.
- Les sorties historiques contenant des facteurs manquants remplacés par `1` restent des runs de diagnostic.

## Performance / run-time

Comparaison seq vs parallèle:

- `examples/STAR/results/ate_10_40_parallel_speed_test.json`

Conclusion:

- `proc4` est globalement le plus rapide,
- `threads4` n'apporte pas de gain robuste,
- écarts d'ATE mineurs possibles (MC/numérique) entre modes.
