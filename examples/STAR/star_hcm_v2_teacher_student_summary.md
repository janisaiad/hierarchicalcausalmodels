# STAR — HCM v2 teacher/class -> student

- **Outcome analysé** : `M` = maths (`gktmathss`) (intervention toujours sur `Q^a`).

## Schéma

- unité `i` : classe / enseignant (`gktchid`)
- sous-unité `j` : élève dans la classe `i`
- variable latente de niveau unité : `U`
- variable observée de niveau unité : `S = gksurban`
- traitement de niveau élève : `A_ij = 1{small class}`
- outcome de niveau élève ciblé par ce run : variable `M` (maths (`gktmathss`))
- régresseurs de niveau élève :
  - `M_ij = gktmathss`
  - `G_ij = 1{female}`
  - `E_ij = 1{white/asian}`
  - `L_ij = 1{free lunch}`

Cette v2 garde les variables du notebook de causal discovery pour rester comparable, mais les replace à un niveau plus fidèle à `STAR` : `schoolk` au niveau classe et les covariables individuelles au niveau élève.

## Choix des régresseurs

- Keep the same discovery notebook variables for comparability.
- Move only school urbanicity to the unit level.
- Keep math, gender, ethnicity, and lunch at the student level.
- Use binary codings for categorical student regressors to stabilize first-pass HCM estimation.

## Données

- source : `examples/STAR/STA-207/STAR_Students.tab`
- grade : `kindergarten`
- classes utilisées comme unités : `322`
- élèves retenus par classe : `10`
- lignes après suppression des valeurs manquantes et filtrage des classes valides : `5745`

## Résultats

| Graph | Status | Identifiable | ATE | E[do(1)] | E[do(0)] |
|---|---|---:|---:|---:|---:|
| `PC` | `ok` | yes | 0.0000 | 485.9134 | 485.9134 |
| `FCI` | `ok` | yes | 0.0000 | 485.9134 | 485.9134 |
| `DirectLiNGAM` | `ok` | yes | 22.2657 | 1382.5388 | 1360.2730 |
| `ExactBIC` | `ok` | yes | 39.9642 | 2406.6516 | 2366.6874 |
| `ConsensusMean` | `ok` | yes | 0.0000 | 485.9134 | 485.9134 |

## Lecture

Cette v2 est meilleure que la v1 au niveau du schéma hiérarchique, car elle colle mieux au mécanisme réel du traitement. En revanche, les graphes de départ restent issus d'une causal discovery plate, puis traduits dans un HCM : cette étape reste donc un choix de modélisation, pas une vérité causale directement lue dans `STAR`.

Pour `PC`, `FCI` et `ConsensusMean`, la formule identifiée se réduit souvent à `$P(Q^m)$` (marginalisation sans effet net de `do(Q^a)`). Un `ATE = 0` dans ce cas reflète surtout la structure graphique, pas une preuve d'absence d'effet de la petite classe.

L'estimateur numérique aligne désormais strictement les régresseurs de conditionnement entre l'ajustement et l'évaluation (même ensemble de parents `Q` que dans la formule, même règle de colonne pour les profils 2D). Les graphes denses (`DirectLiNGAM`, `ExactBIC`) peuvent toutefois rester bruyants ou sensibles à la paramétrisation.

### PC

- note : PC + orientation heuristique des arêtes non orientées.
- statut : `ok`
- formule : $P\left(Qm\right)$
- explication : Do-calculus identification succeeded.

### FCI

- note : FCI squelette seulement; orientation heuristique imposée.
- statut : `ok`
- formule : $P\left(Qm\right)$
- explication : Do-calculus identification succeeded.

### DirectLiNGAM

- note : Arcs dirigés repris de DirectLiNGAM.
- statut : `ok`
- formule : $\sum_{Qe,Qg,Ql_e,Qm_a_e_g_l,S}{P\left(S\mid Qe\right) \cdot P\left(Qm_a_e_g_l\right) \cdot P\left(Ql_e\mid S\right) \cdot P\left(Qm\mid Qa,Qe,Qg,Ql_e,Qm_a_e_g_l\right) \cdot P\left(Qg\right) \cdot P\left(Qe\right)}$
- explication : Do-calculus identification succeeded.

### ExactBIC

- note : DAG score-based repris de ExactBIC.
- statut : `ok`
- formule : $\sum_{Qe,Qg,Qm_e_y,Qy_a_g,S}{P\left(S\mid Qe\right) \cdot P\left(Qy_a_g\mid S\right) \cdot P\left(Qm_e_y\right) \cdot P\left(Qm\mid Qa,Qe,Qg,Qm_e_y,Qy_a_g\right) \cdot P\left(Qg\right) \cdot P\left(Qe\right)}$
- explication : Do-calculus identification succeeded.

### ConsensusMean

- note : Graphe moyen majoritaire de `preliminar_results.md`.
- statut : `ok`
- formule : $P\left(Qm\right)$
- explication : Do-calculus identification succeeded.
