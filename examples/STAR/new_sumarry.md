# Synthèse STAR — version alignée outcome maths (M)

## Objectif

Cette synthèse regroupe uniquement les éléments alignés sur l'outcome maths `M` (`gktmathss`).

## Baseline math (référence économétrique)

Fichiers:

- `examples/STAR/results/star_baseline_math_benchmark.json`
- `examples/STAR/star_baseline_math_benchmark.md`

Valeurs principales:

- OLS `small` autour de `8.10` à `9.50` selon la spécification.
- 2SLS (`class size`) `-1.2186` point de score maths par élève.

## HCM et interprétation

- Les résultats HCM doivent être interprétés à outcome constant (`M`).
- Les runs historiques avec facteurs manquants (remplacés par `1`) restent utiles comme diagnostic, pas comme estimate causal final.

## Facteurs et graphes effectifs

- Analyse facteurs:
  - `examples/STAR/results/ate_10_40_full_artifacts_20260415T075215Z/factor_analysis_historic_17_21`
- Graphes effectifs:
  - `examples/STAR/figures/factor_missing_graphs/DirectLiNGAM_with_factor1_effective_graph_curved.png`
  - `examples/STAR/figures/factor_missing_graphs/ExactBIC_with_factor1_effective_graph_curved.png`

## Speed benchmark (seq vs parallèle)

Source:

- `examples/STAR/results/ate_10_40_parallel_speed_test.json`

Conclusion opérationnelle:

- `proc4` est généralement le mode le plus rapide,
- `threads4` peut être neutre ou plus lent,
- légères différences numériques possibles entre modes.

## Vérifications complémentaires

- Distribution d'hétérogénéité genre par classe:
  - `examples/STAR/figures/score_distribution_diagnostics/class_gender_heterogeneity_distribution.png`
  - `examples/STAR/figures/score_distribution_diagnostics/class_gender_heterogeneity_values.csv`
