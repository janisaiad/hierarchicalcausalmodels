# STAR — HCM v2 teacher/class -> student

## Schéma

- unité `i` : classe / enseignant (`gktchid`)
- sous-unité `j` : élève dans la classe `i`
- variable latente de niveau unité : `U`
- variable observée de niveau unité : `S = gksurban`
- traitement de niveau élève : `A_ij = 1{small class}`
- outcome de niveau élève : `Y_ij = gktreadss`
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
| `PC` | `ok` | yes | 0.0000 | 436.8804 | 436.8804 |
| `FCI` | `ok` | yes | 0.0000 | 436.8804 | 436.8804 |
| `DirectLiNGAM` | `error` | yes |  |  |  |
| `ExactBIC` | `error` | yes |  |  |  |
| `ConsensusMean` | `ok` | yes | 0.0000 | 436.8804 | 436.8804 |

## Lecture

Cette v2 est meilleure que la v1 au niveau du schéma hiérarchique, car elle colle mieux au mécanisme réel du traitement. En revanche, les graphes de départ restent issus d'une causal discovery plate, puis traduits dans un HCM : cette étape reste donc un choix de modélisation, pas une vérité causale directement lue dans `STAR`.

Le point le plus important est que, pour `PC`, `FCI` et `ConsensusMean`, la formule identifiée se réduit à `$P(Qy)$`. Autrement dit, dans ces traductions HCM, l'intervention sur `Q^a` disparaît de l'expression finale. Le `ATE = 0.0000` obtenu ici ne doit donc pas être lu comme une absence d'effet crédible de `stark`, mais comme le signe que cette spécification rend l'effet interventionnel trivial dans ces graphes.

Pour `DirectLiNGAM` et `ExactBIC`, l'identification symbolique réussit, mais l'étape d'estimation casse avec une incohérence du nombre de features dans les régressions internes. Il s'agit donc d'un blocage technique de l'estimateur actuel sur ces factorisations, pas d'un résultat causal substantiel.

### PC

- note : PC + orientation heuristique des arêtes non orientées.
- statut : `ok`
- formule : $P\left(Qy\right)$
- explication : Do-calculus identification succeeded.

### FCI

- note : FCI squelette seulement; orientation heuristique imposée.
- statut : `ok`
- formule : $P\left(Qy\right)$
- explication : Do-calculus identification succeeded.

### DirectLiNGAM

- note : Arcs dirigés repris de DirectLiNGAM.
- statut : `error`
- formule : $\sum_{Qe,Qg,Ql_e,Qm_a_e_g_l,Qy_g_l_m,S}{P\left(S\mid Qe\right) \cdot P\left(Qy_g_l_m\right) \cdot P\left(Qm_a_e_g_l\right) \cdot P\left(Ql_e\mid S\right) \cdot P\left(Qy\mid Qa,Qe,Qg,Ql_e,Qm_a_e_g_l,Qy_g_l_m\right) \cdot P\left(Qg\right) \cdot P\left(Qe\right)}$
- erreur : `ValueError: X has 6 features, but LinearRegression is expecting 4 features as input.`

### ExactBIC

- note : DAG score-based repris de ExactBIC.
- statut : `error`
- formule : $\sum_{Qe,Qg,Qy_a_g,S}{P\left(S\mid Qe\right) \cdot P\left(Qy_a_g\mid S\right) \cdot P\left(Qy\mid Qa,Qg,Qy_a_g\right) \cdot P\left(Qg\right) \cdot P\left(Qe\right)}$
- erreur : `ValueError: X has 3 features, but LinearRegression is expecting 2 features as input.`

### ConsensusMean

- note : Graphe moyen majoritaire de `preliminar_results.md`.
- statut : `ok`
- formule : $P\left(Qy\right)$
- explication : Do-calculus identification succeeded.
