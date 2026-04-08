# Résumé linéaire de l'analyse causale sur STAR

L'analyse a commencé par une mise à plat du problème causal dans `STAR` autour de la question centrale suivante : quel est l'effet du type de classe `stark` sur les performances en lecture `readk` en maternelle, et comment replacer cette question dans une structure hiérarchique fidèle aux données. La première étape a consisté à travailler sur une représentation plate du jeu de données, à partir de `data/star/STAR.csv`, avec les variables `stark`, `readk`, `mathk`, `gender`, `ethnicity`, `lunchk` et `schoolk`.

Dans `star_causal_discovery.ipynb` et `star_causal_discovery.py`, les variables catégorielles ont été encodées en entiers afin de rendre possible l'utilisation d'algorithmes de découverte causale qui supposent des entrées numériques. Cette transformation a été explicitement traitée comme une approximation méthodologique, et non comme une représentation parfaite de la structure causale réelle. Une phase d'exploration descriptive a ensuite été menée avec des histogrammes, un nuage de points `readk` contre `mathk`, une matrice de corrélation sur les variables encodées et une visualisation de `readk` selon `stark`. Cette étape a servi à caractériser la distribution des données avant l'apprentissage des graphes.

La causal discovery proprement dite a ensuite été lancée avec plusieurs méthodes : `PC`, `FCI`, `DirectLiNGAM` et `ExactBIC`, avec des essais complémentaires via `CDT` lorsque cela était possible. Les sorties ont été harmonisées sous forme d'arêtes orientées et de squelettes, puis comparées entre elles à l'aide d'indices de recouvrement de type Jaccard. Un graphe de consensus a aussi été construit à partir des accords observés entre méthodes. Les résultats ont montré qu'il existait des régularités plausibles autour de variables comme `mathk`, `lunchk`, `ethnicity` ou `schoolk`, mais qu'en revanche le statut causal de `stark` n'était pas retrouvé de manière nette ni stable dans cette représentation plate. Cette tension a été interprétée comme un point méthodologique central : le design expérimental de `STAR` donne une crédibilité causale forte à `stark`, mais cette information ne se laisse pas simplement reconstruire par découverte causale sur une table encodée i.i.d.

En parallèle, `star_explainability.ipynb` et `star_explainability.py` ont ajouté une lecture prédictive et explicative. Ici, l'objectif n'était plus d'identifier un graphe causal, mais d'étudier comment un modèle prédictif moderne utilise les variables disponibles pour prédire `readk`. Un pipeline avec encodage one-hot et `RandomForestRegressor` a été entraîné sur les variables catégorielles de maternelle, en incluant notamment `schoolidk` en plus de `schoolk` et `stark`. Les performances prédictives ont ensuite été complétées par plusieurs outils d'explicabilité : importance par permutation, `SHAP`, `LIME`, ainsi que des courbes `PDP` et `ICE` pour `stark`.

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
