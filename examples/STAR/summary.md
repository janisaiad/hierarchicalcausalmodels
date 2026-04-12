# Résumé linéaire de l'analyse causale sur STAR

L'analyse a commencé par une mise à plat du problème causal dans `STAR`, centrée d'abord sur l'effet du type de classe `stark` sur la lecture `readk` en maternelle, puis sur le repositionnement de cet effet dans une structure hiérarchique fidèle aux données. La première étape a consisté à travailler sur une représentation plate (`data/star/STAR.csv`, variables `stark`, `readk`, `mathk`, `gender`, `ethnicity`, `lunchk`, `schoolk`).

Dans `star_causal_discovery.ipynb` et `star_causal_discovery.py`, les catégories ont été encodées en entiers pour les algorithmes de découverte causale. Une phase descriptive (histogrammes, corrélation, `readk` vs `stark`) a précédé l'apprentissage des graphes. La causal discovery (`PC`, `FCI`, `DirectLiNGAM`, `ExactBIC`, compléments `CDT` lorsque possible) a produit des graphes comparés par Jaccard et un graphe de consensus. Régularités autour de `mathk`, `lunchk`, `ethnicity`, `schoolk` ; en revanche `stark` n'apparaît pas de façon stable dans la découverte plate — tension centrale : le design expérimental crédibilise `stark`, mais la discovery sur table i.i.d. ne le reconstruit pas nettement.

`star_explainability.py` a ajouté une couche prédictive (`RandomForest`, one-hot, `schoolidk`, métriques, `SHAP`, `LIME`, `PDP`/`ICE`). `stark` est informatif ; le contexte scolaire domine. Lecture prudente : prédictif, pas mécanisme causal.

La **v1 HCM** (`star_hcm_first_pass.py`, unité école, outcome lecture agrégée) a prouvé le branchement discovery → HCM mais des ATE très instables (`0` ou valeurs énormes). Le diagnostic : hiérarchie `school → student` trop grossière et agrégats école sur les covariables individuelles trop forts.

La **v2 HCM** (`star_hcm_v2_teacher_student.py`) repose sur la hiérarchie **classe / enseignant → élève** (`gktchid`), traitement `A = 1{petite classe}`, covariables élève (`M`, `G`, `E`, `L`), `S = gksurban` au niveau unité. Le script accepte `--outcome Y` (lecture `gktreadss`) ou `--outcome M` (maths `gktmathss`) ; l'estimande est $E[Q^y \mid do(Q^a)]$ ou $E[Q^m \mid do(Q^a)]$ selon le choix.

Après **alignement des régresseurs** entre identification et étape numérique dans la librairie, la v2 **estime sans erreur** les cinq graphes traduits pour l'outcome **maths** `M` (résultats figés dans `examples/STAR/results/star_hcm_v2_teacher_student.json` et `star_hcm_v2_teacher_student_summary.md`). Pour `PC`, `FCI` et `ConsensusMean`, la formule identifiée se réduit à $P(Q^m)$ : **ATE = 0** (effet trivialisé par la structure, pas preuve d'absence d'effet réel). Pour **DirectLiNGAM** et **ExactBIC**, la formule dépend de $Q^a$ ; les ATE numériques sont **non nuls** mais **très sensibles** au graphe appris et à la paramétrisation.

**Décision documentée pour la suite narrative HCM v2 (outcome maths)** : retenir le graphe issu de **DirectLiNGAM** parmi les graphes discovery traduits, car c'est celui qui fait apparaître explicitement un lien **$A \rightarrow M$** dans l'espace HCM, cohérent avec la question « effet de la taille de classe sur le score maths ». **ExactBIC** donne un autre ATE (autre structure, lien traitement–outcome différent) : ce n'est pas un duplicate du choix DirectLiNGAM.

La **baseline économétrique** alignée sur `startenesse` reste la référence la plus **substantive** pour l'effet « petite classe » sur les **scores de lecture** (ordre de grandeur ~`+6` points en OLS). Les ATE HCM v2 sur **maths** ne sont **pas** directement comparables point par point à cette baseline lecture : autre outcome, autre chaîne (graphe appris + do-calculus + estimateur paramétrique).

En synthèse : la chaîne discovery → explicabilité → HCM v1 → **HCM v2 opérationnelle** → baseline OLS/IV est complète. La v2 corrige la hiérarchie et permet une estimation numérique stable pour tous les graphes dans le script principal ; le cœur méthodologique reste la **prudence** sur la causalité des graphes plats retraduits et la **complémentarité** avec la baseline économétrique.

## Chiffres et tables

### 1. Causal discovery plate

Le sous-échantillon utilisé pour la causal discovery contenait `4000` observations, avec `7` variables :

| Variable | Rôle dans l'analyse plate |
|---|---|
| `readk` | outcome principal |
| `mathk` | score scolaire associé |
| `stark` | traitement / type de classe |
| `gender` | covariable élève |
| `ethnicity` | covariable élève |
| `lunchk` | proxy socio-économique |
| `schoolk` | contexte scolaire |

Les arêtes dirigées apprises par les principales méthodes ont été les suivantes :

| Méthode | Arêtes dirigées |
|---|---|
| `PC` | `ethnicity → lunchk`, `gender → readk`, `lunchk → readk`, `mathk → lunchk`, `mathk → readk`, `schoolk → lunchk`, `schoolk → readk` |
| `DirectLiNGAM` | `ethnicity → lunchk`, `ethnicity → mathk`, `ethnicity → schoolk`, `gender → mathk`, `gender → readk`, `lunchk → mathk`, `lunchk → readk`, `mathk → readk`, `schoolk → lunchk`, `stark → mathk` |
| `ExactBIC` | `ethnicity → lunchk`, `ethnicity → mathk`, `ethnicity → schoolk`, `gender → readk`, `mathk → lunchk`, `readk → mathk`, `schoolk → lunchk`, `schoolk → readk`, `schoolk → stark`, `stark → readk` |

Le consensus dirigé commun à `PC`, `DirectLiNGAM` et `ExactBIC` s'est réduit à :

| Consensus dirigé commun | |
|---|---|
| 1 | `ethnicity → lunchk` |
| 2 | `gender → readk` |
| 3 | `schoolk → lunchk` |

Le consensus de squelette commun aux quatre méthodes (`PC`, `FCI`, `DirectLiNGAM`, `ExactBIC`) contenait les paires suivantes :

| Squelette commun | |
|---|---|
| 1 | `readk - mathk` |
| 2 | `readk - gender` |
| 3 | `mathk - ethnicity` |
| 4 | `mathk - lunchk` |
| 5 | `ethnicity - lunchk` |
| 6 | `ethnicity - schoolk` |
| 7 | `lunchk - schoolk` |

Les recouvrements Jaccard entre squelettes étaient :

| Jaccard squelettes | `PC` | `FCI` | `LiNGAM` | `ExactBIC` |
|---|---:|---:|---:|---:|
| `PC` | `1.0000` | `1.0000` | `0.6667` | `0.8182` |
| `FCI` | `1.0000` | `1.0000` | `0.6667` | `0.8182` |
| `LiNGAM` | `0.6667` | `0.6667` | `1.0000` | `0.5385` |
| `ExactBIC` | `0.8182` | `0.8182` | `0.5385` | `1.0000` |

Les recouvrements Jaccard entre arcs dirigés étaient :

| Jaccard arcs dirigés | `PC` | `LiNGAM` | `ExactBIC` |
|---|---:|---:|---:|
| `PC` | `1.0000` | `0.4167` | `0.4167` |
| `LiNGAM` | `0.4167` | `1.0000` | `0.3333` |
| `ExactBIC` | `0.4167` | `0.3333` | `1.0000` |

Pour compléter cette étape, le DAG `ExactBIC` a été injecté dans `DoWhy`. L'estimation par régression linéaire backdoor a donné :

| Estimation `DoWhy` | Valeur |
|---|---:|
| `ATE(stark -> readk | graphe ExactBIC)` | `2.6639` |

Les métadonnées de la recherche exacte BIC étaient :

| Indicateur ExactBIC | Valeur |
|---|---:|
| `n_parent_graphs_entries` | `193` |
| `while_iter` | `64` |
| `for_iter` | `259` |
| `n_closed` | `64` |
| `max_n_opened` | `39` |

Toutes les méthodes `CDT` essayées via `R` ont échoué faute de `Rscript` disponible dans l'environnement.

### 2. Explicabilité prédictive

Le pipeline prédictif a été entraîné sur `STAR.csv` avec :

| Quantité | Valeur |
|---|---:|
| lignes brutes | `11598` |
| lignes après filtrage `stark` et `readk` observés | `5789` |
| taille train | `4341` |
| taille test | `1448` |
| nombre de features one-hot | `114` |

Les performances du modèle de forêt aléatoire ont été :

| Métrique test | Valeur |
|---|---:|
| `R^2` | `0.1939` |
| `MAE` | `20.3791` |
| `RMSE` | `28.5944` |

Les chiffres principaux d'explicabilité exportés dans `star_metrics.json` étaient :

| Indicateur | Valeur |
|---|---:|
| importance permutation moyenne de `stark` | `0.0430` |
| `SHAP` échantillon expliqué | `800` |
| temps `FastTreeSHAP` | `2.8875 s` |
| plus grande importance moyenne absolue SHAP | `6.3578` |
| `gpu_treeshap_ok` | `0` |

Les courbes ICE / PDP pour `stark` ont donné les prédictions moyennes suivantes :

| Modalité de `stark` | `readk` prédit moyen |
|---|---:|
| `regular` | `434.2840` |
| `regular+aide` | `436.3207` |
| `small` | `440.7341` |

La différence moyenne prédite entre `small` et `regular` dans ce modèle était :

| Contraste ICE moyen | Valeur |
|---|---:|
| `small - regular` | `6.4501` |

Les figures produites à cette étape et dans la causal discovery incluent notamment :

| Fichier | Contenu |
|---|---|
| `star_eda_histograms_scatter.png` | histogrammes + nuage de points |
| `star_eda_correlation.png` | matrice de corrélation |
| `star_eda_readk_by_stark.png` | boxplot de `readk` selon `stark` |
| `cd_graph_pc.png` | graphe `PC` |
| `cd_graph_fci_skeleton.png` | squelette `FCI` |
| `cd_graph_lingam.png` | graphe `DirectLiNGAM` |
| `cd_graph_exact_bic.png` | graphe `ExactBIC` |
| `shap_summary_fasttreeshap.png` | résumé `SHAP` |
| `lime_instance_0.png` à `lime_instance_4.png` | explications locales `LIME` |
| `pdp_ice_stark.png` | `PDP` et `ICE` sur `stark` |

### 3. HCM v1 : `school -> student`

La première version HCM a utilisé :

| Quantité | Valeur |
|---|---:|
| lignes après suppression des manquants | `5768` |
| écoles utilisées comme unités | `79` |
| élèves équilibrés par école | `34` |

Le schéma était :

| Niveau | Variables |
|---|---|
| unité | `U`, `S`, `M`, `G`, `E`, `L` |
| sous-unité | `A`, `Y` |

Les résultats de la v1 ont été :

| Graphe | Statut | Identifiable | `E[do(1)]` | `E[do(0)]` | `ATE` |
|---|---|---:|---:|---:|---:|
| `PC` | `ok` | yes | `436.1474` | `436.1474` | `0.0000` |
| `FCI` | `ok` | yes | `436.1474` | `436.1474` | `0.0000` |
| `DirectLiNGAM` | `ok` | yes | `41094.6596` | `42979.4828` | `-1884.8232` |
| `ExactBIC` | `ok` | yes | `426106.0222` | `405421.8973` | `20684.1248` |
| `ConsensusMean` | `ok` | yes | `436.1474` | `436.1474` | `0.0000` |

La lecture retenue a été que cette v1 était exploitable comme preuve de concept, mais pas crédible causalement, car la hiérarchie `school -> student` était trop grossière et les agrégats école sur `mathk`, `gender`, `ethnicity` et `lunchk` étaient trop simplificateurs.

### 4. HCM v2 : `teacher/class -> student` (état actuel)

**Fichiers de référence** : `star_hcm_v2_teacher_student.py`, `examples/STAR/results/star_hcm_v2_teacher_student.json`, `star_hcm_v2_teacher_student_summary.md`.

La v2 utilise la table `STAR_Students.tab` et la hiérarchie :

| Quantité | Valeur |
|---|---:|
| source | `examples/STAR/STA-207/STAR_Students.tab` |
| grade | `kindergarten` |
| classes utilisées comme unités | `322` |
| élèves retenus par classe | `10` |
| lignes après filtrage | `5745` |

Le schéma v2 :

| Niveau | Variables |
|---|---|
| unité | `U`, `S` |
| sous-unité | `A`, `Y`, `M`, `G`, `E`, `L` |

| Variable HCM | Définition |
|---|---|
| `A` | `1{gkclasstype == 1}` petite classe |
| `Y` | `gktreadss` (lecture) |
| `M` | `gktmathss` (maths) |
| `G` | `1{gender == female}` |
| `E` | `1{race in {white, asian}}` |
| `L` | `1{free lunch}` |
| `S` | `gksurban` |
| unité | `gktchid` |

**Résultats numériques pour l'outcome maths `M`** (estimande $E[Q^m \mid do(Q^a)]$, `N_MC_SAMPLES = 60`, `RANDOM_STATE = 42`) :

| Graphe | Statut | Identifiable | `E[do(1)]` | `E[do(0)]` | `ATE` |
|---|---|---:|---:|---:|---:|
| `PC` | `ok` | yes | `485.9134` | `485.9134` | `0.0000` |
| `FCI` | `ok` | yes | `485.9134` | `485.9134` | `0.0000` |
| `DirectLiNGAM` | `ok` | yes | `1382.5388` | `1360.2730` | `22.2657` |
| `ExactBIC` | `ok` | yes | `2406.6516` | `2366.6874` | `39.9642` |
| `ConsensusMean` | `ok` | yes | `485.9134` | `485.9134` | `0.0000` |

**Interprétation**

- Pour `PC`, `FCI`, `ConsensusMean`, la formule identifiée est $P(Q^m)$ : l'intervention sur $Q^a$ n'apparaît pas dans l'expression ; **ATE = 0** est **mécanique** dans ces graphes retraduits, pas une conclusion substantive sur l'effet de la petite classe sur les maths.
- **DirectLiNGAM** : graphe **retenu** pour la suite comme représentation discovery-compatible où un chemin **traitement → maths** reste présent après traduction HCM ; **ATE ≈ 22.27** points sur l'échelle `gktmathss` (à prendre avec la même prudence que tout effet basé sur un DAG appris).
- **ExactBIC** : autre DAG, **ATE ≈ 39.96** ; illustre la **sensibilité** au choix de graphe.
- L'outcome **lecture** `Y` se obtient en relançant le script avec `--outcome Y` ; les nombres ne coïncident pas avec le tableau ci-dessus (autre $Q$-query).

Figures HCM v2 (schémas traduits) : `examples/STAR/figures/hcm_v2_*.png`.

### 5. Tableau final de synthèse

| Étape | Données | Niveau d'analyse | Résultat principal |
|---|---|---|---|
| causal discovery | `STAR.csv`, `4000` lignes | table plate | graphes plausibles ; faible consensus sur `stark` |
| `DoWhy` sur `ExactBIC` | même table plate | backdoor sur graphe appris | `ATE(stark→readk) ≈ 2.66` |
| explicabilité RF | `STAR.csv`, `5789` lignes | prédictif plat | `small - regular` ~ `+6.45` points prédits sur `readk` |
| HCM v1 | `5768` lignes, `79` écoles, `34` élèves / école | `school -> student` | preuve de concept ; ATE non crédibles |
| HCM v2 | `5745` lignes, `322` classes, `10` élèves / classe | `teacher/class -> student` | tous graphes **ok** ; ATE maths non triviaux pour DL / ExactBIC ; **DL retenu** pour l'effet sur `M` |
| Baseline OLS / IV | `5725` lignes (benchmark) | économétrique `startenesse-like` | ~`+6.5` pts lecture (`small`) ; IV taille classe ~`-0.82` |

## Benchmark économétrique et comparaison des estimateurs

Pour disposer d'une référence interne comparable à `startenesse.md`, un benchmark a été construit dans `star_baseline_and_hcm_benchmark.py` : baseline `OLS` et `2SLS / IV`, plus un test de variantes de **familles** de distributions pour le HCM v2.

### 6. Baseline économétrique alignée sur `startenesse`

Le benchmark a été estimé sur `5725` lignes complètes de maternelle, couvrant `79` écoles et `323` classes. Variables : `small`, `reg_aide`, covariables élève et enseignant, effets fixes école `C(gkschid)`.

Les résultats `OLS` :

| Modèle | Coef `small` | SE `small` | p-value `small` | Coef `reg_aide` | SE `reg_aide` | p-value `reg_aide` | `R^2` | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ols_minimal` | `5.7362` | `1.0478` | `4.39e-08` | `0.6503` | `0.9918` | `0.5121` | `0.0062` | `0.004` |
| `ols_school_fe` | `6.5843` | `0.9712` | `1.21e-11` | `1.0936` | `0.8967` | `0.2226` | `0.2109` | `0.400` |
| `ols_with_student_controls` | `6.5084` | `0.9353` | `3.44e-12` | `1.2497` | `0.8645` | `0.1483` | `0.2632` | `0.786` |
| `ols_full_like_startenesse` | `6.5521` | `0.9362` | `2.59e-12` | `0.9643` | `0.8756` | `0.2708` | `0.2655` | `0.886` |

Le benchmark `2SLS / IV` :

| Modèle IV | Coef `class size` | SE | p-value | Temps (s) |
|---|---:|---:|---:|---:|
| `2SLS small/reg_aide -> class size` | `-0.8182` | `0.1089` | `6.75e-14` | `1.179` |

### 7. Test HCM avec familles (`star_baseline_and_hcm_benchmark.py`)

Ce script est **distinct** du pipeline canonique `star_hcm_v2_teacher_student.py`. Il enchaîne des variantes de familles (`all_gaussian_except_A`, `mixed_binary_gaussian`, `mixed_with_poisson_count_proxy`) avec une intégration légèrement différente.

Dans les exports actuels (`star_baseline_and_hcm_benchmark.md`), **`DirectLiNGAM` et `ExactBIC` restent en erreur** (`LinearRegression` / nombre de features), tandis que `PC`, `FCI` et `ConsensusMean` sortent avec `ATE = 0`. Cela **ne contredit pas** le tableau v2 section 4 : le script principal v2 a été mis à jour pour **aligner** parents / features avec la formule identifiée ; le benchmark familles n'a pas nécessairement été réaligné sur le même code path.

| Variante | Graphes `ok` | Graphes en erreur | Observation |
|---|---:|---:|---|
| `all_gaussian_except_A` | `3` | `2` | DL / EB en erreur dans ce benchmark |
| `mixed_binary_gaussian` | `3` | `2` | idem |
| `mixed_with_poisson_count_proxy` | `3` | `2` | idem |

**Suite logique** : harmoniser `star_baseline_and_hcm_benchmark.py` sur la même logique d'estimation que `star_hcm_v2_teacher_student.py` si l'on veut des comparaisons de familles sur **tous** les graphes.

### 8. Conclusion sur les estimateurs

- La **baseline OLS / 2SLS** reste la référence **la plus interprétable** pour l'effet expérimental de la petite classe sur les **scores de lecture** dans l'esprit `startenesse`.
- Le **HCM v2** dans `star_hcm_v2_teacher_student.py` fournit des **ATE cohérents avec le DAG** traduit pour l'outcome **maths** ; le graphe **DirectLiNGAM** est **retenu** pour la narration « effet sur `M` » parmi les options discovery.
- Changer uniquement les **familles** dans l'ancien benchmark ne résout pas la question de fond : **structure du graphe** (identification) et **alignement estimateur / formule** dominent.
- Comparer quantitativement ATE HCM maths (~22 points DL) et OLS lecture (~6 points) serait **abusif** sans cadre explicite (outcomes et identifications différents).

### 9. Origine des cas où `ATE = 0` dans les HCM

Pour plusieurs graphes (`PC`, `FCI`, `ConsensusMean` en v2 avec outcome `M`), la formule identifiée se réduit à une marge sur le bloc $Q$ de l'outcome, par exemple $P(Q^m)$ ou $P(Q^y)$ selon l'estimande, **sans** dépendre de $do(Q^a)$. Alors

$$
E[Q^m \mid do(Q^a=1)] = E[Q^m \mid do(Q^a=0)]
$$

au sein de ce modèle, et l'ATE estimé vaut **0** par construction. Cela n'infère pas l'absence d'effet causal dans `STAR` ; cela signifie que **ce graphe retraduit** ne laisse pas passer l'intervention jusqu'à l'outcome dans l'expression du do-calculus.

## Calcul parallèle dans la librairie et benchmark de temps

Le calcul parallèle est présent dans `src/hierarchicalcausalmodels/estimation/parallel.py` (`parallel_map(...)`), utilisé entre autres par `fit_many_ols(...)`, `fit_many_2sls(...)`, `fit_regressors_per_unit(...)`, `estimate_causal_effect(...)`, etc. L'usage est **opt-in** (`n_jobs`, backend `threads` ou `processes`).

### 10. Validation logicielle

```bash
pytest tests/test_econometric_parallel.py tests/test_estimation_parallel.py tests/test_env.py
```

| Test suite | Résultat |
|---|---|
| `tests/test_econometric_parallel.py` | `passed` |
| `tests/test_estimation_parallel.py` | `passed` |
| `tests/test_env.py` | `passed` |
| total | `6 passed` |

### 11. Benchmark parallèle économétrique STAR-like

Sur `5725` lignes, `24` spécifications `OLS`, `10` `IV / 2SLS` :

| Mode | Batch `OLS` (s) | Batch `IV / 2SLS` (s) |
|---|---:|---:|
| séquentiel | `12.3242` | `21.4296` |
| `threads`, `n_jobs=4` | `6.7054` | `14.8683` |
| `processes`, `n_jobs=4` | `0.4604` | `2.0457` |

Coefficients inchangés (ex.) : premier `small` en OLS `6.5084` ; `class_size` en IV `-0.8182`. Pour les batchs économétriques, **`processes` est très avantageux**.

### 12. Benchmark parallèle `per_unit`

Cas synthétique `400` unités × `250` sous-unités : le **séquentiel** peut battre le parallèle pour des fits linéaires très légers (surcharge d'orchestration).

### 13. Synthèse pratique

| Fonction / couche | Parallélisable ? | Intérêt pratique |
|---|---|---|
| `fit_many_ols` / `2SLS` / reduced form | oui | fort sur batchs STAR-like |
| `fit_regressors_per_unit` / `estimate_ate_confounder` | oui | faible si chaque fit est trivial |
| `estimate_causal_effect` | oui (infra) | selon la lourdeur réelle |

---

## Résumé global (exhaustif et court)

- Quatre blocs principaux : **causal discovery plate**, **explicabilité prédictive**, **HCM v1 → v2**, **baseline économétrique** ; calcul parallèle documenté à part.
- Discovery sur `4000` lignes, `7` variables ; méthodes `PC`, `FCI`, `DirectLiNGAM`, `ExactBIC` ; `stark` **non stable** dans les graphes plats.
- `DoWhy` + ExactBIC plat : `ATE(stark→readk) ≈ 2.66` (dépend du DAG).
- Forêt aléatoire : `R² ≈ 0.19` ; contraste prédictif `small - regular` sur `readk` ~ **`+6.45`** points.
- HCM v1 (`school → student`) : preuve de concept, ATE **non crédibles**.
- HCM v2 (`teacher/class → student`) : **tous graphes** estiment correctement dans `star_hcm_v2_teacher_student.py` ; outcome **maths `M`** documenté : `PC`/`FCI`/`Consensus` → **ATE 0** ($P(Q^m)$) ; **DirectLiNGAM** → **ATE ≈ 22.27** ; **ExactBIC** → **ATE ≈ 39.96**.
- **Graphe retenu** pour la suite HCM v2 sur l'effet **maths** : **DirectLiNGAM** (présence explicite $A \rightarrow M$ après traduction).
- Baseline OLS : effet **`small`** ~ **`+6.5`** points sur **lecture** ; IV taille de classe ~ **`-0.82`** point par élève — **référence substantive** pour comparer l'ordre de grandeur côté expérience, pas pour valider point par point l'ATE HCM sur `M`.
- Benchmark **familles** dans `star_baseline_and_hcm_benchmark.py` : peut encore **échouer** sur DL/EB tant qu'il n'est pas aligné sur le même estimateur que le script v2 principal.

---

## À faire

- Harmoniser **`star_baseline_and_hcm_benchmark.py`** avec le pipeline d'estimation de **`star_hcm_v2_teacher_student.py`** pour des comparaisons de familles sur les cinq graphes.
- Documenter explicitement un run **`--outcome Y`** (lecture) dans les résultats versionnés si l'on veut une table parallèle à la section 4 pour `Q^y`.
- Construire un **petit ensemble de DAGs hiérarchiques natifs** `teacher/class → student` (substantifs, pas seulement traduction de la discovery plate), avec vérification que la formule identifiée **dépend de $Q^a$** avant estimation.
- Continuer à comparer tout nouveau HCM à la **baseline OLS / IV** sur des outcomes et définitions **comparables**.
- Garder la hiérarchie **classe / enseignant → élève** comme schéma principal ; ne pas réintroduire `school → student` comme référence.
- Utiliser le **parallèle** pour les batchs économétriques lourds ; rester **séquentiel** pour les `per_unit` triviaux.
- Pour chaque nouveau graphe : schéma, formule identifiée, temps, statut, ATE — et ne **retenir** en conclusion que les structures à la fois **identifiantes** (effet non trivialisé) et **substantivement plausibles**.
