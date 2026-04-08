# Résumé linéaire de l'analyse causale sur STAR

L'analyse a commencé par une mise à plat du problème causal dans `STAR`, centré sur l'effet du type de classe `stark` sur les performances en lecture `readk` en maternelle, puis sur le repositionnement de cet effet dans une structure hiérarchique fidèle aux données. La première étape a consisté à travailler sur une représentation plate du jeu de données, à partir de `data/star/STAR.csv`, avec les variables `stark`, `readk`, `mathk`, `gender`, `ethnicity`, `lunchk` et `schoolk`.

Dans `star_causal_discovery.ipynb` et `star_causal_discovery.py`, les variables catégorielles ont été encodées en entiers afin de rendre possible l'utilisation d'algorithmes de découverte causale qui supposent des entrées numériques. Cette transformation a été explicitement traitée comme une approximation méthodologique, et non comme une représentation parfaite de la structure causale réelle. Une phase d'exploration descriptive a ensuite été menée avec des histogrammes, un nuage de points `readk` contre `mathk`, une matrice de corrélation sur les variables encodées et une visualisation de `readk` selon `stark`. Cette étape a servi à caractériser la distribution des données avant l'apprentissage des graphes.

La causal discovery proprement dite a ensuite été lancée avec plusieurs méthodes : `PC`, `FCI`, `DirectLiNGAM` et `ExactBIC`, avec des essais complémentaires via `CDT` lorsque cela était possible. Les sorties ont été harmonisées sous forme d'arêtes orientées et de squelettes, puis comparées entre elles à l'aide d'indices de recouvrement de type Jaccard. Un graphe de consensus a aussi été construit à partir des accords observés entre méthodes. Les résultats ont montré qu'il existait des régularités plausibles autour de variables comme `mathk`, `lunchk`, `ethnicity` ou `schoolk`, mais qu'en revanche le statut causal de `stark` n'était pas retrouvé de manière nette ni stable dans cette représentation plate. Cette tension a été interprétée comme un point méthodologique central : le design expérimental de `STAR` donne une crédibilité causale forte à `stark`, mais cette information ne se laisse pas simplement reconstruire par découverte causale sur une table encodée i.i.d.

En parallèle, `star_explainability.ipynb` et `star_explainability.py` ont ajouté une lecture prédictive et explicative. Cette étape visait à étudier comment un modèle prédictif moderne utilise les variables disponibles pour prédire `readk`, sans chercher à identifier directement un graphe causal. Un pipeline avec encodage one-hot et `RandomForestRegressor` a été entraîné sur les variables catégorielles de maternelle, en incluant notamment `schoolidk` en plus de `schoolk` et `stark`. Les performances prédictives ont ensuite été complétées par plusieurs outils d'explicabilité : importance par permutation, `SHAP`, `LIME`, ainsi que des courbes `PDP` et `ICE` pour `stark`.

Cette étape d'explicabilité a confirmé que `stark` porte bien du signal prédictif dans le modèle, mais elle a aussi montré que le contexte scolaire, en particulier `schoolidk` et `schoolk`, structure fortement la prédiction. L'interprétation retenue a été prudente : ces outils documentent le comportement d'un modèle prédictif, pas le vrai mécanisme causal de `STAR`. Néanmoins, ils ont servi de préliminaire utile en montrant que la dépendance de `readk` à l'environnement scolaire et social est suffisamment forte pour justifier une modélisation hiérarchique explicite.

La première tentative de passage à un modèle hiérarchique causal a ensuite été réalisée dans `star_hcm_first_pass.py`, puis synthétisée dans `star_hcm_first_pass.md` et `star_hcm_first_pass_summary.md`. L'idée a été de repartir des graphes appris dans la causal discovery et de les traduire dans le cadre des `Hierarchical Causal Models`. Pour rendre le problème estimable dans le framework existant, un premier schéma hiérarchique a été choisi avec `schoolidk` comme unité et les élèves comme sous-unités. Le traitement restait `A_ij = 1{stark == small}` et l'outcome restait `Y_ij = readk`, mais les autres variables, comme `mathk`, `gender`, `ethnicity` et `lunchk`, ont été agrégées au niveau école sous forme de moyennes ou de proportions.

Cette v1 HCM a rempli son rôle de preuve de concept. Elle a montré qu'il était effectivement possible de brancher les graphes issus de la causal discovery sur le pipeline HCM, de construire un schéma hiérarchique opérationnel, puis de lancer l'identification et l'estimation de l'effet causal. Cependant, les résultats obtenus ont été très instables. Pour plusieurs graphes, l'ATE estimé valait numériquement `0.0000`, tandis que d'autres graphes, notamment `ExactBIC`, donnaient des valeurs extrêmement grandes et peu crédibles substantiellement. La synthèse rédigée dans `star_hcm_first_pass_summary.md` a conclu que cette première passe était utile comme preuve de connexion entre la discovery plate et le moteur HCM, mais pas encore comme analyse causale crédible de `STAR`.

Le diagnostic posé après cette première passe a été clair. Le problème principal ne venait plus du fait de réussir ou non à exécuter le pipeline, mais du choix de hiérarchie et du choix des niveaux de variables. La hiérarchie `school -> student` a été jugée trop grossière pour représenter correctement le mécanisme réel du traitement `stark`, qui opère beaucoup plus naturellement au niveau classe / enseignant. De plus, le fait de transformer des variables individuelles comme `mathk`, `gender`, `ethnicity` et `lunchk` en agrégats école a été considéré comme une simplification trop forte. La conclusion a donc été qu'il fallait passer à une version plus fidèle où l'unité hiérarchique serait la classe ou l'enseignant, et non plus l'école.

Cette refonte a conduit à `star_hcm_v2_teacher_student.py`, puis à `star_hcm_v2_teacher_student_summary.md`. Dans cette v2, la hiérarchie a été redéfinie de manière plus proche de `STAR` : l'unité est désormais la classe / l'enseignant, identifiée par `gktchid`, et la sous-unité est l'élève dans cette classe. Le traitement est défini à ce niveau comme petite classe ou non, l'outcome devient `gktreadss`, la variable scolaire `gksurban` est placée au niveau unité, et les covariables `math`, `gender`, `race` et `free lunch` restent au niveau élève. Cette architecture est conceptuellement plus cohérente avec le traitement réellement attribué dans le protocole STAR.

Les graphes issus de la causal discovery initiale ont de nouveau été traduits dans l'espace HCM, puis utilisés pour lancer une seconde série d'identifications et d'estimations. Cette v2 a été considérée comme meilleure que la v1 du point de vue du schéma causal hiérarchique. Cependant, elle n'a pas encore produit une analyse causale pleinement satisfaisante. Pour `PC`, `FCI` et `ConsensusMean`, l'identification se réduit à une formule du type `$P(Qy)$`, ce qui signifie que l'intervention sur `Q^a` disparaît de l'expression finale. Le fait d'obtenir un `ATE = 0.0000` dans ces cas n'a donc pas été interprété comme une preuve d'absence d'effet causal de `stark`, mais comme le signe que, dans ces traductions HCM particulières, l'effet interventionnel devient trivial. Pour `DirectLiNGAM` et `ExactBIC`, l'identification symbolique réussit, mais l'estimation échoue ensuite à cause d'un problème technique de cohérence du nombre de variables dans les régressions internes.

Au total, la chaîne d'analyse causale sur `STAR` raconte un cheminement méthodologique progressif. On est d'abord parti d'une causal discovery plate sur une table encodée, qui a fourni des graphes hypothétiques et mis en évidence l'instabilité des liens autour de `stark`. On a ensuite ajouté une couche d'explicabilité prédictive pour comprendre où se situait le signal dans les données et confirmer l'importance du contexte scolaire. Puis on a tenté un premier branchement HCM au niveau `school -> student`, qui a fonctionné comme preuve de concept mais s'est révélé trop grossier. Enfin, on a construit une v2 plus fidèle avec la hiérarchie `teacher/class -> student`, qui améliore nettement la structure du modèle mais ne donne pas encore une estimation causale stable et convaincante.

La conclusion générale est donc que le projet a déjà franchi plusieurs étapes importantes : la découverte causale exploratoire, l'audit prédictif, la connexion au pipeline HCM et la correction d'une première erreur de hiérarchie. En revanche, la question causale n'est pas encore résolue de manière satisfaisante. La suite la plus naturelle consiste soit à construire des graphes hiérarchiques plus nativement adaptés à `STAR`, soit à corriger les limites actuelles de l'estimation HCM pour les factorisations plus complexes issues de `DirectLiNGAM` et `ExactBIC`.

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

La lecture retenue a été que cette v1 était exploitable comme preuve de concept, mais pas crédible causalement, car :
- la hiérarchie `school -> student` était trop grossière
- les agrégats école sur `mathk`, `gender`, `ethnicity` et `lunchk` étaient trop simplificateurs

### 4. HCM v2 : `teacher/class -> student`

La v2 a utilisé la table `STAR_Students.tab` et la hiérarchie :

| Quantité | Valeur |
|---|---:|
| source | `examples/STAR/STA-207/STAR_Students.tab` |
| grade | `kindergarten` |
| classes utilisées comme unités | `322` |
| élèves retenus par classe | `10` |
| lignes après filtrage | `5745` |

Le schéma v2 était :

| Niveau | Variables |
|---|---|
| unité | `U`, `S` |
| sous-unité | `A`, `Y`, `M`, `G`, `E`, `L` |

Le mapping des variables était :

| Variable HCM | Définition |
|---|---|
| `A` | `1{gkclasstype == 1}` |
| `Y` | `gktreadss` |
| `M` | `gktmathss` |
| `G` | `1{gender == female}` |
| `E` | `1{race in {white, asian}}` |
| `L` | `1{free lunch}` |
| `S` | `gksurban` |
| unité | `gktchid` |

Les résultats de la v2 ont été :

| Graphe | Statut | Identifiable | `E[do(1)]` | `E[do(0)]` | `ATE` |
|---|---|---:|---:|---:|---:|
| `PC` | `ok` | yes | `436.8804` | `436.8804` | `0.0000` |
| `FCI` | `ok` | yes | `436.8804` | `436.8804` | `0.0000` |
| `DirectLiNGAM` | `error` | yes |  |  |  |
| `ExactBIC` | `error` | yes |  |  |  |
| `ConsensusMean` | `ok` | yes | `436.8804` | `436.8804` | `0.0000` |

Les erreurs techniques en v2 étaient :

| Graphe | Erreur |
|---|---|
| `DirectLiNGAM` | `ValueError: X has 6 features, but LinearRegression is expecting 4 features as input.` |
| `ExactBIC` | `ValueError: X has 3 features, but LinearRegression is expecting 2 features as input.` |

L'interprétation de la v2 a été plus fine que celle de la v1. Pour `PC`, `FCI` et `ConsensusMean`, la formule identifiée se réduit à `$P(Qy)$`, ce qui signifie que l'intervention disparaît dans l'expression estimée. Le `ATE = 0.0000` ne doit donc pas être lu comme une absence d'effet causal crédible de `stark`, mais comme une trivialisation de l'effet dans ces traductions HCM spécifiques.

### 5. Tableau final de synthèse

| Étape | Données | Niveau d'analyse | Résultat principal |
|---|---|---|---|
| causal discovery | `STAR.csv`, `4000` lignes | table plate | graphes plausibles mais faible consensus direct sur `stark` |
| `DoWhy` sur `ExactBIC` | même table plate | ajustement backdoor sur graphe appris | `ATE = 2.6639` |
| explicabilité RF | `STAR.csv`, `5789` lignes | prédictif plat | `stark` informatif, mais poids fort du contexte scolaire |
| HCM v1 | `5768` lignes, `79` écoles, `34` élèves / école | `school -> student` | preuve de concept, ATE très instables |
| HCM v2 | `5745` lignes, `322` classes, `10` élèves / classe | `teacher/class -> student` | schéma meilleur, mais estimation encore non stabilisée |

## Benchmark économétrique et comparaison des estimateurs

Pour disposer d'une vraie référence interne comparable à `startenesse.md`, un benchmark supplémentaire a été construit dans `star_baseline_and_hcm_benchmark.py`. L'objectif était double. D'une part, reproduire une baseline économétrique de type `OLS` et `2SLS / IV` avec les mêmes niveaux de variables, les mêmes contrôles et le même esprit que la réplication économétrique. D'autre part, relancer le HCM v2 avec plusieurs choix de familles de distributions afin de tester si une meilleure adéquation entre la nature des variables et les estimateurs pouvait stabiliser les résultats.

### 6. Baseline économétrique alignée sur `startenesse`

Le benchmark a été estimé sur `5725` lignes complètes de maternelle, couvrant `79` écoles et `323` classes. Les variables utilisées suivent directement la logique de `startenesse` :

- traitement principal : `small`
- groupe `regular+aide` : `reg_aide`
- covariables élève : `white_asian`, `female`, `free_lunch`
- covariables enseignant : `white_teacher`, `gktyears`, `masters_plus`
- effets fixes école : `C(gkschid)`

Les résultats `OLS` sont les suivants :

| Modèle | Coef `small` | SE `small` | p-value `small` | Coef `reg_aide` | SE `reg_aide` | p-value `reg_aide` | `R^2` | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ols_minimal` | `5.7362` | `1.0478` | `4.39e-08` | `0.6503` | `0.9918` | `0.5121` | `0.0062` | `0.004` |
| `ols_school_fe` | `6.5843` | `0.9712` | `1.21e-11` | `1.0936` | `0.8967` | `0.2226` | `0.2109` | `0.400` |
| `ols_with_student_controls` | `6.5084` | `0.9353` | `3.44e-12` | `1.2497` | `0.8645` | `0.1483` | `0.2632` | `0.786` |
| `ols_full_like_startenesse` | `6.5521` | `0.9362` | `2.59e-12` | `0.9643` | `0.8756` | `0.2708` | `0.2655` | `0.886` |

Ces coefficients sont cohérents avec l'ordre de grandeur de `startenesse.md`, où l'effet d'une petite classe sur les scores est typiquement d'environ `5` à `8` points selon les spécifications et les niveaux scolaires. Ici, le signal est très stable : être dans une petite classe est associé à un gain d'environ `+6.5` points sur le score de lecture, tandis que l'effet de `regular+aide` reste faible et non robuste.

Le benchmark `2SLS / IV` a ensuite utilisé `small` et `reg_aide` comme instruments de la taille réalisée de classe `gkclasssize`, avec les mêmes contrôles et effets fixes école. Le résultat est :

| Modèle IV | Coef `class size` | SE | t-stat | p-value | Temps (s) |
|---|---:|---:|---:|---:|---:|
| `2SLS small/reg_aide -> class size` | `-0.8182` | `0.1089` | `-7.5116` | `6.75e-14` | `1.179` |

L'interprétation de ce coefficient est classique : une augmentation d'un élève dans la classe est associée à une baisse d'environ `0.82` point du score, ce qui est très cohérent avec la lecture économétrique de `startenesse`.

### 7. Test HCM avec familles mieux choisies

Le benchmark HCM a repris la v2 `teacher/class -> student`, mais en testant plusieurs variantes de familles de distributions pour éviter de tout traiter comme gaussien par défaut.

Les trois variantes testées ont été :

| Variante | Familles |
|---|---|
| `all_gaussian_except_A` | `A` en `bernoulli`, tout le reste en `gaussian` |
| `mixed_binary_gaussian` | `A`, `G`, `E`, `L` en `bernoulli`; `Y`, `M`, `S` en `gaussian` |
| `mixed_with_poisson_count_proxy` | comme ci-dessus, mais `S` en `poisson` comme stress test |

Les résultats ont été :

| Variante | Graphe | Statut | `ATE` | Temps (s) |
|---|---|---|---:|---:|
| `all_gaussian_except_A` | `PC` | `ok` | `0.0000` | `0.100` |
| `all_gaussian_except_A` | `FCI` | `ok` | `0.0000` | `0.093` |
| `all_gaussian_except_A` | `DirectLiNGAM` | `error` |  | `0.189` |
| `all_gaussian_except_A` | `ExactBIC` | `error` |  | `0.045` |
| `all_gaussian_except_A` | `ConsensusMean` | `ok` | `0.0000` | `0.080` |
| `mixed_binary_gaussian` | `PC` | `ok` | `0.0000` | `0.071` |
| `mixed_binary_gaussian` | `FCI` | `ok` | `0.0000` | `0.070` |
| `mixed_binary_gaussian` | `DirectLiNGAM` | `error` |  | `0.481` |
| `mixed_binary_gaussian` | `ExactBIC` | `error` |  | `0.059` |
| `mixed_binary_gaussian` | `ConsensusMean` | `ok` | `0.0000` | `0.139` |
| `mixed_with_poisson_count_proxy` | `PC` | `ok` | `0.0000` | `0.073` |
| `mixed_with_poisson_count_proxy` | `FCI` | `ok` | `0.0000` | `0.073` |
| `mixed_with_poisson_count_proxy` | `DirectLiNGAM` | `error` |  | `0.780` |
| `mixed_with_poisson_count_proxy` | `ExactBIC` | `error` |  | `0.055` |
| `mixed_with_poisson_count_proxy` | `ConsensusMean` | `ok` | `0.0000` | `0.083` |

Les erreurs techniques sur `DirectLiNGAM` et `ExactBIC` persistent :

| Graphe | Erreur |
|---|---|
| `DirectLiNGAM` | `ValueError: X has 6 features, but LinearRegression is expecting 4 features as input.` |
| `ExactBIC` | `ValueError: X has 3 features, but LinearRegression is expecting 2 features as input.` |

Le classement automatique des variantes HCM a donné :

| Variante | Graphes `ok` | Graphes en erreur | Écart médian absolu à `5.34` | ATE absolu moyen | Temps total (s) |
|---|---:|---:|---:|---:|---:|
| `all_gaussian_except_A` | `3` | `2` | `5.3400` | `0.0000` | `0.507` |
| `mixed_binary_gaussian` | `3` | `2` | `5.3400` | `0.0000` | `0.820` |
| `mixed_with_poisson_count_proxy` | `3` | `2` | `5.3400` | `0.0000` | `1.065` |

Il faut toutefois bien interpréter ce classement. La variante `all_gaussian_except_A` est classée première non parce qu'elle fournit un meilleur effet causal, mais simplement parce qu'elle n'est pas pire que les autres et qu'elle est un peu plus rapide. Sur le fond, les trois variantes HCM échouent de la même manière : elles gardent `ATE = 0.0000` pour les graphes qui tournent, et elles cassent sur `DirectLiNGAM` et `ExactBIC`.

### 8. Conclusion actualisée sur les estimateurs

Le résultat important de ce benchmark est donc le suivant.

La baseline économétrique alignée sur `startenesse` fonctionne bien et fournit une référence interne crédible :

- effet `small` autour de `+6.5` points
- effet `regular+aide` faible et non robuste
- effet `IV` sur la taille de classe d'environ `-0.82` point par élève

À l'inverse, le HCM actuel ne devient pas convaincant simplement en changeant les familles de distributions. Le problème de fond ne semble donc pas être seulement un mauvais choix entre `gaussian`, `bernoulli` ou `poisson`, mais plus profondément :

- la traduction des graphes plats en HCM
- la structure des factorisations identifiées
- et la robustesse de l'estimateur sur ces formules

On peut donc dire, à ce stade, que la meilleure référence quantitative disponible dans le projet est désormais la baseline économétrique `OLS` / `2SLS`, et non les HCM testés jusqu'ici.

### 9. Origine des cas où `ATE = 0` dans les HCM

Un point important à expliciter est que le `ATE = 0.0000` observé pour plusieurs graphes HCM, en particulier `PC`, `FCI` et `ConsensusMean`, ne signifie pas que les données `STAR` concluent à une absence d'effet causal des petites classes.

La raison est structurelle. Dans ces cas, après traduction du graphe plat en HCM puis identification par do-calculus, la formule causale obtenue se réduit à une expression du type :

$$
P(Qy)
$$

Autrement dit, l'intervention sur `Q^a` disparaît complètement de la formule finale. Dès lors, les deux quantités

$$
E[Q^y \mid do(Q^a = 1)]
\quad \text{et} \quad
E[Q^y \mid do(Q^a = 0)]
$$

deviennent identiques dans cette spécification, et leur différence vaut mécaniquement zéro.

Il faut donc lire ce `ATE = 0` comme un **effet trivialisé par le modèle**, et non comme un **effet réellement nul dans STAR**. Ce résultat signifie seulement que, dans ces graphes retraduits en HCM, la structure causale identifiée ne laisse plus apparaître l'intervention comme déterminante dans l'expression finale de l'outcome.

Cette situation est cohérente avec le diagnostic général du projet :

- les graphes appris par causal discovery sont des graphes plats
- ils ne sont pas appris nativement dans un cadre hiérarchique
- leur traduction en HCM peut donc perdre ou écraser la structure interventionnelle utile

Cela explique pourquoi la baseline économétrique alignée sur `startenesse` donne un effet positif robuste des petites classes, alors que plusieurs HCM traduits donnent `ATE = 0`.

## Calcul parallèle dans la librairie et benchmark de temps

Le calcul parallèle est désormais présent explicitement dans la librairie. Une couche de parallélisation commune est disponible dans `src/hierarchicalcausalmodels/estimation/parallel.py`, autour de la fonction `parallel_map(...)`. Cette infrastructure est utilisée par plusieurs estimateurs et aides au calcul, notamment :

- `fit_many_ols(...)`
- `fit_many_reduced_form(...)`
- `fit_many_2sls(...)`
- `fit_per_unit_estimators(...)`
- `fit_regressors_per_unit(...)`
- `estimate_ate_confounder(...)`
- `ast_to_estimator(...)`
- `estimate_causal_effect(...)`

Le point important est que cette parallélisation est **opt-in** et non automatique. Autrement dit, la librairie sait faire du calcul parallèle, mais il faut l'activer explicitement avec `n_jobs > 1`, et choisir un backend comme `threads` ou `processes`.

### 10. Validation logicielle

Les tests spécifiques à cette nouvelle couche économétrique parallèle ont été relancés. La commande exécutée a été :

```bash
pytest tests/test_econometric_parallel.py tests/test_estimation_parallel.py tests/test_env.py
```

Résultat :

| Test suite | Résultat |
|---|---|
| `tests/test_econometric_parallel.py` | `passed` |
| `tests/test_estimation_parallel.py` | `passed` |
| `tests/test_env.py` | `passed` |
| total | `6 passed` |

Cette étape valide que les résultats numériques sont cohérents entre exécution séquentielle et exécution parallèle.

### 11. Benchmark parallèle sur la couche économétrique STAR-like

Un benchmark a été réalisé sur :

- `5725` lignes
- `24` spécifications `OLS`
- `10` spécifications `IV / 2SLS`

La comparaison de temps d'exécution a porté sur :

- mode séquentiel
- `threads` avec `n_jobs = 4`
- `processes` avec `n_jobs = 4`

Les résultats obtenus sont :

| Mode | Batch `OLS` (s) | Batch `IV / 2SLS` (s) |
|---|---:|---:|
| séquentiel | `12.3242` | `21.4296` |
| `threads`, `n_jobs=4` | `6.7054` | `14.8683` |
| `processes`, `n_jobs=4` | `0.4604` | `2.0457` |

Les coefficients estimés restent identiques dans tous les cas :

| Quantité contrôlée | Valeur |
|---|---:|
| premier coefficient `small` en `OLS` | `6.5084` |
| premier coefficient `class_size` en `IV` | `-0.8182` |

Le résultat principal est donc que, pour la **couche économétrique STAR-like en batch**, le calcul parallèle est réellement utile, et que le backend `processes` est ici beaucoup plus rapide que `threads`.

### 12. Benchmark parallèle sur la couche `per_unit`

Le même type de comparaison a été mené sur la couche `per_unit`, avec un cas synthétique de taille :

- `400` unités
- `250` sous-unités par unité

Les fonctions testées ont été :

- `fit_regressors_per_unit(...)`
- `estimate_ate_confounder(...)`

Les résultats sont :

| Mode | `fit_regressors_per_unit` (s) | `estimate_ate_confounder` (s) |
|---|---:|---:|
| séquentiel | `0.1443` | `0.1342` |
| `threads`, `n_jobs=4` | `0.2538` | `0.2624` |
| `processes`, `n_jobs=4` | `0.2277` | `0.2447` |

Les résultats causaux restent identiques :

| Quantité contrôlée | Valeur |
|---|---:|
| nombre de régressions ajustées | `400` |
| ATE synthétique estimé | `0.4504` |

Mais ici, l'effet sur le temps est inverse : le séquentiel est plus rapide que le parallèle. Cela signifie que, pour des tâches unitaires très légères comme des régressions linéaires simples par unité, le coût d'orchestration du parallèle dépasse le gain de calcul.

### 13. Interprétation pratique

Le benchmark montre donc que le calcul parallèle n'a pas le même intérêt partout.

Pour la **couche économétrique de batch** :

- le parallélisme apporte un gain net
- le backend `processes` est le plus performant dans ce benchmark
- la cohérence numérique des coefficients est conservée entre séquentiel et parallèle

Pour la **couche `per_unit` simple** :

- la parallélisation est disponible
- elle n'est pas rentable sur de petits fits linéaires
- le mode séquentiel reste le plus efficace tant que chaque tâche unitaire est très rapide

On peut donc résumer la situation ainsi :

| Fonction / couche | Parallélisable ? | Intérêt pratique observé |
|---|---|---|
| `fit_many_ols(...)` | oui | fort |
| `fit_many_2sls(...)` | oui | fort |
| `fit_many_reduced_form(...)` | oui | fort probable par analogie |
| `fit_regressors_per_unit(...)` | oui | faible sur petits modèles |
| `estimate_ate_confounder(...)` | oui | faible sur petits modèles |
| `estimate_causal_effect(...)` | oui en infrastructure | dépend de la lourdeur réelle des sous-tâches |

La conclusion pratique est donc la suivante : la librairie contient bien maintenant une infrastructure de calcul parallèle de base, mais son utilisation optimale dépend du type de charge. Pour des batchs économétriques STAR-like, le gain est réel et important. Pour des estimateurs par unité très rapides, le séquentiel reste souvent préférable.








resume global exhaustif et court :

- Le travail a relié quatre blocs : causal discovery plate, explicabilité prédictive, HCM v1, HCM v2, puis baseline économétrique de référence.
- La causal discovery a été faite sur `4000` observations et `7` variables : `readk`, `mathk`, `stark`, `gender`, `ethnicity`, `lunchk`, `schoolk`.
- Les méthodes principales utilisées dans la discovery ont été `PC`, `FCI`, `DirectLiNGAM` et `ExactBIC`.
- Les liens les plus stables retrouvés dans les graphes plats concernent surtout `ethnicity`, `lunchk`, `gender`, `readk` et `schoolk`.
- Le traitement `stark` n’a pas été retrouvé de manière robuste et consensuelle dans la causal discovery plate.
- Le graphe `ExactBIC` injecté dans `DoWhy` a donné un `ATE(stark -> readk)` d’environ `2.6639`, mais ce résultat dépend fortement du graphe appris.
- La couche d’explicabilité prédictive a utilisé une forêt aléatoire sur `5789` lignes avec encodage one-hot.
- Les performances prédictives obtenues sont modérées : `R^2 = 0.1939`, `MAE = 20.3791`, `RMSE = 28.5944`.
- Dans le modèle prédictif, la modalité `small` est associée à environ `+6.45` points par rapport à `regular`, mais cette lecture reste prédictive et non causale.
- Le premier HCM v1 a utilisé la hiérarchie `school -> student`, avec `79` écoles et `34` élèves équilibrés par école.
- Les résultats v1 sont très instables : `ATE = 0` pour `PC`, `FCI` et `ConsensusMean`, `ATE = -1884.8232` pour `DirectLiNGAM`, `ATE = 20684.1248` pour `ExactBIC`.
- Le diagnostic principal de la v1 est que la hiérarchie `school -> student` est trop grossière et que les agrégats école sur les covariables sont trop simplificateurs.
- Le HCM v2 a corrigé la structure hiérarchique en passant à `teacher/class -> student`, avec `322` classes et `10` élèves retenus par classe.
- Dans la v2, `PC`, `FCI` et `ConsensusMean` donnent encore `ATE = 0.0000`, tandis que `DirectLiNGAM` et `ExactBIC` échouent à l’étape d’estimation.
- Le `ATE = 0` dans plusieurs HCM ne signifie pas absence d’effet causal de `stark`, mais disparition de l’intervention dans la formule identifiée, qui se réduit à `P(Qy)`.
- Une baseline économétrique alignée sur `startenesse` a été ajoutée pour fournir une vraie référence interne comparable.
- Cette baseline donne un effet `small` stable autour de `+6` à `+6.6` points selon les spécifications `OLS`.
- Le `2SLS / IV` donne un effet de taille de classe d’environ `-0.8182` point par élève, cohérent avec la littérature STAR-like.
- Cette baseline économétrique constitue actuellement la référence la plus crédible du projet, bien plus que les HCM testés jusqu’ici.
- Plusieurs variantes HCM avec familles mieux choisies ont été testées (`bernoulli`, `gaussian`, `poisson`), sans amélioration substantielle des résultats.
- Le problème principal des HCM actuels semble donc venir moins du choix de famille que de la traduction des graphes plats en graphes hiérarchiques et de la robustesse de l’estimation.
- Les graphes hiérarchiques effectivement impliqués sont de deux types : `school -> student` pour la v1 et `teacher/class -> student` pour la v2.
- Le calcul parallèle est maintenant intégré dans la librairie via `parallel_map(...)` et les nouvelles fonctions économétriques parallélisables.
- Les tests de cette couche parallèle passent (`6 passed`) et valident la cohérence numérique entre exécution séquentielle et parallèle.
- Sur des batchs économétriques STAR-like, le parallèle est très utile, surtout avec le backend `processes`.
- Sur des estimateurs `per_unit` très simples et rapides, le séquentiel reste plus efficace que le parallèle.
- L’état actuel du projet est donc le suivant : la chaîne complète fonctionne, la hiérarchie pertinente est mieux comprise, la baseline économétrique est solide, mais l’estimation HCM causale n’est pas encore stabilisée.
- La suite logique consiste à construire des graphes hiérarchiques nativement plausibles pour `STAR`, puis à comparer leurs résultats directement à la baseline `OLS / IV`.




a faire : 

- construire un petit ensemble de graphes hiérarchiques natifs `teacher/class -> student`, justifiés directement par la structure substantielle de `STAR` et par les résultats de `startenesse`
- partir de graphes simples où l’effet du traitement reste explicitement présent, avec en priorité des structures contenant `U -> A`, `U -> Y`, `S -> A`, `S -> Y`, `A -> Y` et `E, G, L, M -> Y`
- tester éventuellement une variante avec médiation partielle du type `A -> M -> Y`, mais sans multiplier trop tôt les structures complexes
- vérifier systématiquement, avant l’estimation numérique, que la formule identifiée dépend réellement de `Q^a`
- écarter les graphes pour lesquels l’identification se réduit à `P(Qy)`, car ils trivialisent l’intervention et conduisent mécaniquement à `ATE = 0`
- conserver la hiérarchie `teacher/class -> student` comme base de travail, et ne plus utiliser `school -> student` comme schéma principal
- garder la baseline économétrique `OLS` / `2SLS` alignée sur `startenesse` comme benchmark interne de référence
- comparer chaque nouveau HCM à cette baseline, en vérifiant que l’ordre de grandeur des effets reste compatible avec un gain d’environ `+6` points pour `small`
- corriger les erreurs techniques de l’estimateur sur les factorisations complexes, en particulier les problèmes de dimensions rencontrés sur `DirectLiNGAM` et `ExactBIC`
- auditer la construction des matrices de features dans l’estimateur, notamment la cohérence entre parents identifiés, ordre des covariables, ajustement et prédiction
- continuer à utiliser des familles adaptées au type de variables : `gaussian` pour `Y` et `M`, `bernoulli/logistique` pour `A`, `G`, `E`, `L`
- éviter de considérer le choix de famille comme la source principale du problème, puisque plusieurs variantes ont déjà été testées sans amélioration substantielle
- utiliser le calcul parallèle pour accélérer les batchs de spécifications économétriques et les comparaisons de modèles
- garder le séquentiel pour les estimateurs `per_unit` très rapides, tant que la surcharge du parallèle reste supérieure au gain de calcul
- documenter chaque nouveau graphe testé avec son schéma, sa formule identifiée, son temps de calcul, son statut d’estimation et son ATE
- retenir à la fin uniquement les graphes hiérarchiques qui gardent un effet interventionnel explicite, un comportement numérique stable et une cohérence substantielle avec la littérature STAR
