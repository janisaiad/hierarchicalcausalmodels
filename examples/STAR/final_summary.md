# Final Summary — STAR HCM (ATE, facteurs, formules)

## Objectif

Ce document résume l'analyse complète des runs historiques STAR HCM autour de :

- l'estimation de l'ATE,
- les facteurs de formule manquants remplacés par `1`,
- l'effet causal/pratique de cette neutralisation,
- la lecture des graphes effectifs.

Les runs ciblés ici sont ceux reproduits à l'identique :

- `DirectLiNGAM`: `ATE = 17.618368229358794`
- `ExactBIC`: `ATE = 21.628105518204393`

avec `HCM_DISABLE_MULTIPARENT_Q_PRECOMPUTE=1`.

## Formulation papier de l'ATE (rappel)

Dans le formalisme HCM, l'effet moyen est :

$$
\mathrm{ATE} = \mathbb{E}[Y \mid do(A=1)] - \mathbb{E}[Y \mid do(A=0)].
$$

Dans la forme générale d'identification (produit de facteurs + marginalisation) :

$$
\mathbb{E}[Y \mid do(A=a)]
=
\sum_{\mathbf{q}\in\mathcal{Q}}
\prod_{k} P\!\left(Q_k \mid \mathrm{Pa}(Q_k)\right),
$$

où la somme/intégrale porte sur les variables sommées de la formule identifiée (approchée numériquement par énumération discrète ou Monte Carlo selon les cas).

## Formules identifiées effectivement utilisées pour les runs historiques

### DirectLiNGAM

$$
\sum_{Qe,Qg,Ql_e,Qm_a_e_g_l,Qy_g_l_m,S}{
P(S\mid Qe)\cdot
P(Qy_g_l_m)\cdot
P(Qm_a_e_g_l)\cdot
P(Ql_e\mid S)\cdot
P(Qy\mid Qa,Qe,Qg,Ql_e,Qm_a_e_g_l,Qy_g_l_m)\cdot
P(Qg)\cdot
P(Qe)}
$$

Valeurs :

- $E[do(1)] = 1451.3002598815938$
- $E[do(0)] = 1433.681891652235$
- $\mathrm{ATE} = 17.618368229358794$

### ExactBIC

$$
\sum_{Qe,Qg,Qy_a_g,S}{
P(S\mid Qe)\cdot
P(Qy_a_g\mid S)\cdot
P(Qy\mid Qa,Qg,Qy_a_g)\cdot
P(Qg)\cdot
P(Qe)}
$$

Écriture intégrale complètement décomposée (formalisme papier, variables continues/discrètes traitées via intégrale/somme selon leur nature) :

$$
\mathbb{E}[Y \mid do(Q^a=\delta_{a_\star})]
=
\int \Bigg[
\int y \; q^y(y)\, \mathrm{d}y
\Bigg]
\; p\!\left(q^y \mid q^a=\delta_{a_\star}, q^g, q^{y|a,g}\right)
\; p\!\left(q^{y|a,g}\mid s\right)
\; p(s\mid q^e)
\; p(q^g)\,p(q^e)\,
\mathrm{d}q^y\,\mathrm{d}q^{y|a,g}\,\mathrm{d}s\,\mathrm{d}q^g\,\mathrm{d}q^e .
$$

Avec l'ATE :

$$
\mathrm{ATE}
=
\mathbb{E}[Y \mid do(Q^a=\delta_{1})]
-\mathbb{E}[Y \mid do(Q^a=\delta_{0})].
$$

Dans l'implémentation courante, le facteur `P(Qy_a_g \mid S)` est manquant sur ce run historique et remplacé par `1`, donc la fonctionnelle effectivement évaluée devient :

$$
\mathbb{E}[Y \mid do(Q^a=\delta_{a_\star})]_{\text{effectif}}
=
\int \Bigg[
\int y \; q^y(y)\, \mathrm{d}y
\Bigg]
\; p\!\left(q^y \mid q^a=\delta_{a_\star}, q^g, q^{y|a,g}\right)
\; p(s\mid q^e)
\; p(q^g)\,p(q^e)\,
\mathrm{d}q^y\,\mathrm{d}q^{y|a,g}\,\mathrm{d}s\,\mathrm{d}q^g\,\mathrm{d}q^e .
$$

Valeurs :

- $E[do(1)] = 1788.1023959048073$
- $E[do(0)] = 1766.474290386603$
- $\mathrm{ATE} = 21.628105518204393$

### Lecture du plot de factor analysis (focus ExactBIC effectif)

Pour ce run, le plot de facteurs se lit ainsi :

- le terme qui bouge entre `do(1)` et `do(0)` est principalement
  $P(Qy \mid Qa,Qg,Qy_a_g)$,
- le facteur `P(Qy_a_g \mid S)` est plat à `1` (missing estimator),
- les facteurs `P(S|Qe)`, `P(Qg)`, `P(Qe)` restent quasi invariants.

Donc, causalement, la différence do1-do0 est portée presque entièrement par le bloc conditionnel sur `Qy`, alors que la branche `Q^{y|a,g}\!\mid S` est neutralisée numériquement.

## Implémentation numérique dans le code

Le pipeline implémente la logique papier via :

1. extraction de la formule identifiée (`identify_effect`),
2. enrichissement des colonnes `Q` (`enriched`),
3. fit des estimateurs de facteurs (`fitted`),
4. évaluation récursive de la formule (`_eval_formula`) avec marginalisation (somme discrète ou Monte Carlo),
5. moyenne par unité (classe) puis différence `do(1)-do(0)`.

Règle clé de robustesse actuelle :

- si un facteur n'a pas d'estimateur (`est is None`), `_eval_formula` renvoie `1.0` pour ce facteur.

Donc un facteur manquant devient multiplicativement neutre.

## Analyse factorielle — runs 17.618 / 21.628

### Résultat principal

Sur ces runs historiques, la variation `do(1)` vs `do(0)` est majoritairement portée par le terme conditionnel principal sur `Qy` :

- DirectLiNGAM :
  - $P(Qy \mid Qa,Qe,Qg,Ql_e,Qm_a_e_g_l,Qy_g_l_m)$
  - moyenne unitaire do1 `440.1617` vs do0 `434.8255`
  - delta `+5.3362` (ratio `1.01227`)
- ExactBIC :
  - $P(Qy \mid Qa,Qg,Qy_a_g)$
  - moyenne unitaire do1 `440.1534` vs do0 `434.8307`
  - delta `+5.3227` (ratio `1.01224`)

Les autres facteurs sont quasi invariants entre do1/do0 dans ces runs.

### Facteurs manquants remplacés par `1`

- DirectLiNGAM :
  - `P(Qy_g_l_m)` -> `1`
  - `P(Qm_a_e_g_l)` -> `1`
- ExactBIC :
  - `P(Qy_a_g | S)` -> `1`

Interprétation :

- le graphe causal de départ n'est pas réécrit,
- mais ces blocs de formule sont neutralisés numériquement,
- donc l'estimande calculée est une version partiellement amputée de certaines dépendances.

## Effet causal de la neutralisation (liens “effectifs”)

Les figures de graphes effectifs ont été générées ici :

- `examples/STAR/figures/factor_missing_graphs/DirectLiNGAM_with_factor1_effective_graph_curved.png`
- `examples/STAR/figures/factor_missing_graphs/ExactBIC_with_factor1_effective_graph_curved.png`

Les liens marqués rouge (sur le panneau initial) correspondent aux dépendances neutralisées par les facteurs `=1`.

### DirectLiNGAM — dépendances neutralisées (7)

- `_A -> _M`
- `_E -> _M`
- `_G -> _M`
- `_L -> _M`
- `_G -> _Y`
- `_L -> _Y`
- `_M -> _Y`

### ExactBIC — dépendances neutralisées (2)

- `_A -> _Y`
- `_G -> _Y`

## Ce que cela implique pour la confiance dans ces ATE

- Ces ATE (`17.6`, `21.6`) sont bien reproductibles dans la configuration historique.
- Mais ils reposent sur une formule où certains facteurs structurants sont neutralisés.
- Donc la lecture causale doit être prudente : l'estimand numérique n'exploite pas toute la structure initialement identifiée.

En résumé :

- reproductibilité : oui,
- cohérence numérique interne : oui,
- fidélité causale complète à la formule “pleine” : non, tant que des facteurs restent remplacés par `1`.

## Artifacts et traçabilité

Run complet :

- `examples/STAR/results/ate_10_40_full_artifacts_20260415T075215Z`

Par graphe :

- `do1_artifacts.pkl` / `do0_artifacts.pkl`
- versions JSON converties (`.pkl.json` et `.pkl.full.json`)
- `summary.json`

Analyses facteurs :

- `examples/STAR/results/ate_10_40_full_artifacts_20260415T075215Z/factor_analysis_historic_17_21/`

Graphes effectifs :

- `examples/STAR/figures/factor_missing_graphs/`

## Interprétation causale détaillée

### 1) Niveau des variables dans ce HCM

- variables avec préfixe `_` : niveau élève (sous-unité),
- variables sans `_` (par ex. `S`, `U`) : niveau classe/unité,
- `U` : confondeur latent de niveau classe (non observé),
- `A` : traitement (petite classe), `Y` : outcome lecture, `M` : score maths.

### 2) Lecture causale des liens (ExactBIC)

Dans le graphe ExactBIC initial, les dépendances majeures observées incluent :

- `U -> _A` et `U -> _Y` : confounding latent entre assignation au traitement et outcome,
- `S -> _A`, `S -> _Y`, `S -> _L` : contexte école/classe impactant assignation et performances,
- `_A -> _Y` : effet causal direct du traitement sur la lecture,
- `_G -> _Y` : effet du genre sur la lecture,
- `_Y -> _M` : dépendance orientée découverte lecture -> maths,
- `_E -> _L`, `_E -> _M`, `_E -> S` : structure socio-démographique/corrélation orientée.

Dans le graphe ExactBIC effectif (configuration historique `factor=1`) :

- les dépendances portées par `Qy_a_g` sont neutralisées numériquement,
- ce qui correspond ici à la perte effective de `_A -> _Y` et `_G -> _Y` dans le calcul.

Conséquence : l'ATE n'est plus porté par la version complète du mécanisme causal “traitement + covariables vers Y”, mais par la partie restante de la formule.

### 3) Lecture causale des liens (DirectLiNGAM)

Dans DirectLiNGAM initial, les facteurs manquants correspondent aux blocs :

- `Qy_g_l_m` (dépendance de `Y` vis-à-vis de `G,L,M`),
- `Qm_a_e_g_l` (dépendance de `M` vis-à-vis de `A,E,G,L`).

En effectif (facteurs forcés à 1), les dépendances suivantes sont neutralisées :

- vers `M` : `_A -> _M`, `_E -> _M`, `_G -> _M`, `_L -> _M`,
- vers `Y` : `_G -> _Y`, `_L -> _Y`, `_M -> _Y`.

Interprétation : une partie importante des chemins médiés/corrélés impliquant `M` et des covariables élève est retirée de l'estimation numérique.

### 4) Chemins causaux, backdoor et biais résiduel

Idéalement, la formule identifiée ferme les chemins de confusion via la factorisation complète et la marginalisation des variables nécessaires.
Quand des facteurs sont remplacés par `1` :

- certains chemins ne sont plus correctement pondérés,
- la fermeture backdoor implicite de la formule devient incomplète,
- l'ATE obtenu est alors celui d'une fonctionnelle tronquée, pas de l'estimand complet du graphe initial.

Cela peut conduire à :

- sous-ajustement de certaines dépendances,
- sensibilité accrue à quelques termes restants (ici surtout `P(Qy | ...)`),
- interprétation causale affaiblie malgré une bonne reproductibilité numérique.

### 5) Pourquoi on observe quand même un ATE non nul (~17–22)

Même avec des facteurs neutralisés, il reste :

- un terme principal conditionnel sur `Qy` qui varie entre do1/do0,
- une structure de marginalisation encore active,
- des effets de contexte unitaire (moyenne sur classes) qui produisent une différence nette.

Donc :

- oui, l'ATE est stable/reproductible dans ce setup,
- non, il ne représente pas l'effet causal “plein” que donnerait la formule sans facteurs manquants.

### 6) Position d'interprétation recommandée

Pour reporting scientifique :

- présenter ces ATE comme **runs historiques de diagnostic**,
- expliciter que certains facteurs ont été neutralisés (`=1`),
- éviter de conclure causalement au même niveau de confiance qu'un run sans facteurs manquants,
- prioriser les runs avec pré-calcul multi-parents complet pour une lecture causale finale.

## Lecture détaillée des plots d'analyse de facteurs

### Types de plots et rôle

Deux familles de plots ont été produites :

- `*_factor_unit_context_delta.png` (avec `*_factor_unit_context_analysis.json`) :
  c'est le plot le plus informatif pour expliquer l'ATE, car il compare chaque
  facteur entre do1 et do0 avec le contexte unitaire/intervention.
- `*_factor_trace_levels_log10.png` (avec `*_factor_trace_analysis.json`) :
  il montre surtout les niveaux absolus des facteurs (échelle log10), mais ne
  sépare pas bien la contribution do1-do0 dans ces runs.

En pratique pour cette discussion, l'interprétation causale doit s'appuyer en
priorité sur `factor_unit_context_*`.

### Valeurs extraites — ExactBIC

Résultat global :

- `E_do_1 = 1788.1023959048073`
- `E_do_0 = 1766.474290386603`
- `ATE = 21.628105518204393`

Facteurs (`terms_sorted_by_abs_delta_unit`) :

- `P(Qy | Qa, Qg, Qy_a_g)` :
  - do1 `440.1533939755954`
  - do0 `434.83070077346974`
  - delta `+5.322693202125663`
  - ratio `1.012240840383758`
  - `missing_estimator_any = 0`
- `P(S | Qe)` :
  - do1 = do0 = `2.438650306748466`
  - delta `0`
- `P(Qy_a_g | S)` :
  - do1 = do0 = `1.0`
  - delta `0`
  - `missing_estimator_any = 1` (facteur manquant neutralisé)
- `P(Qg)` :
  - do1 = do0 = `0.5080745341614906`
  - delta `0`
- `P(Qe)` :
  - do1 = do0 = `0.6745339813664595`
  - delta `0`

Interprétation ExactBIC :

- le différentiel do1-do0 est quasi entièrement porté par `P(Qy | ...)`,
- `P(Qy_a_g | S)` est forcé à `1` et n'apporte plus de signal causal,
- les autres facteurs restent essentiellement invariants.

### Valeurs extraites — DirectLiNGAM

Résultat global :

- `E_do_1 = 1451.3002598815938`
- `E_do_0 = 1433.681891652235`
- `ATE = 17.618368229358794`

Facteurs (`terms_sorted_by_abs_delta_unit`) :

- `P(Qy | Qa, Qe, Qg, Ql_e, Qm_a_e_g_l, Qy_g_l_m)` :
  - do1 `440.16170729281396`
  - do0 `434.8254944485428`
  - delta `+5.336212844271131`
  - ratio `1.0122720790579187`
  - `missing_estimator_any = 0`
- `P(S | Qe)` :
  - do1 = do0 = `2.438650306748466`
  - delta `0`
- `P(Qy_g_l_m)` :
  - do1 = do0 = `1.0`
  - delta `0`
  - `missing_estimator_any = 1`
- `P(Qm_a_e_g_l)` :
  - do1 = do0 = `1.0`
  - delta `0`
  - `missing_estimator_any = 1`
- `P(Ql_e | S)`, `P(Qg)`, `P(Qe)` :
  - invariants (delta ~ 0)

Interprétation DirectLiNGAM :

- même structure que pour ExactBIC : le terme principal `P(Qy | ...)` porte
  l'essentiel de la variation do1-do0,
- les facteurs multi-parents manquants sont neutralisés à `1` et retirent une
  partie de la structure causale du calcul effectif.

### Comment lire `ExactBIC_factor_trace_levels_log10.png`

Ce plot montre les niveaux absolus en log10 :

- le point le plus à droite (environ `2.6`) correspond au facteur principal
  `P(Qy | Qa,Qg,Qy_a_g)` (valeur absolue élevée),
- `P(Qy_a_g | S)` est autour de `0` en log10 car il vaut `1`,
- les facteurs `P(Qg)` et `P(Qe)` apparaissent à des niveaux plus faibles.

Limite importante :

- ce plot ne suffit pas pour expliquer l'ATE, car il montre des niveaux, pas la
  différence interventionnelle robuste ; pour la contribution causale do1-do0,
  il faut lire `factor_unit_context_delta`.

### Conclusion synthétique sur les facteurs

- Dans les deux graphes, la contribution do1-do0 visible est concentrée sur le
  facteur conditionnel principal de `Qy`.
- Les facteurs manquants (`missing_estimator_any = 1`) sont remplacés par `1`
  et ne contribuent pas au delta.
- Les ATE historiques restent reproductibles numériquement, mais avec une
  interprétation causale partielle tant que ces facteurs manquants ne sont pas
  modélisés/fittés correctement.

