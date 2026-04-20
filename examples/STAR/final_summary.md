# Final Summary — STAR (Math Outcome Only)

## Scope

Ce document est aligné uniquement sur l'outcome maths `M` (`gktmathss`).
Toutes les comparaisons et interprétations ci-dessous concernent les runs maths.

## Baseline économétrique (math)

Source:

- `examples/STAR/results/star_baseline_math_benchmark.json`
- `examples/STAR/star_baseline_math_benchmark.md`

Résultats clés:

- OLS `coef_small`:
  - `ols_minimal`: `8.0962`
  - `ols_school_fe`: `9.4988`
  - `ols_with_student_controls`: `9.3797`
  - `ols_full_like_startenesse`: `9.4316`
- 2SLS `coef_classsize`: `-1.2186` point de score maths par élève

## HCM v2 (math)

La cible HCM v2 est `Q^m` avec intervention sur `Q^a`.

Lecture de principe:

- certains graphes donnent des ATE nuls (fonctionnelle sans dépendance effective à `do(Q^a)`),
- graphes denses (DirectLiNGAM / ExactBIC) peuvent donner des ATE non nuls mais sensibles à la stabilité numérique et à la complétude des facteurs.

## Facteurs manquants et neutralisation

Dans les runs historiques, certains facteurs peuvent être remplacés par `1`.

Conséquence:

- la formule évaluée devient une version tronquée,
- l'interprétation causale est partielle,
- les graphes effectifs doivent être lus comme "dépendances réellement actives dans le calcul".

## Analyse de facteurs (méthode)

Pour expliquer les différences do1/do0:

- privilégier `*_factor_unit_context_delta.png` et `*_factor_unit_context_analysis.json`,
- utiliser `*_factor_trace_levels_log10.png` pour les niveaux absolus, pas pour attribuer la contribution causale do1/do0.

## Speedup séquentiel vs parallèle (runs historiques 10–40)

Source:

- `examples/STAR/results/ate_10_40_parallel_speed_test.json`

Constats:

- `threads4` n'améliore pas systématiquement le temps,
- `proc4` est souvent le meilleur compromis vitesse,
- petites variations numériques d'ATE possibles (ordre du centième) entre modes.

## Artifacts utiles

- Run complet historical: `examples/STAR/results/ate_10_40_full_artifacts_20260415T075215Z`
- Analyse facteurs: `examples/STAR/results/ate_10_40_full_artifacts_20260415T075215Z/factor_analysis_historic_17_21`
- Graphes effectifs: `examples/STAR/figures/factor_missing_graphs`
- Distribution hétérogénéité genre par classe:
  - figure: `examples/STAR/figures/score_distribution_diagnostics/class_gender_heterogeneity_distribution.png`
  - valeurs: `examples/STAR/figures/score_distribution_diagnostics/class_gender_heterogeneity_values.csv`
