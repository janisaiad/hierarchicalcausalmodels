# Specstory — estimateurs parallélisés (implémentation et analyse)

## Contexte

Objectif : récapituler les estimateurs parallélisés implémentés dans la librairie HCM, ce qui a été benchmarké dans STAR, et les conclusions pratiques déjà établies.

## 1) Ce qui est implémenté

### 1.1 Primitive commune

Fichier : `src/hierarchicalcausalmodels/estimation/parallel.py`

- `parallel_map(items, worker_fn, n_jobs=1, backend="threads"|"processes")`
- exécution séquentielle si `n_jobs=1` (ou pas de `joblib`)
- backend opt-in, validation explicite de la valeur

### 1.2 Couche économétrique parallélisée

Fichier : `src/hierarchicalcausalmodels/estimation/econometric_estimators.py`

- `fit_many_ols(...)`
- `fit_many_reduced_form(...)`
- `fit_many_2sls(...)`

Ces fonctions dispatchent en batch via `parallel_map` et conservent les mêmes estimateurs unitaires (`fit_ols`, `fit_2sls`), donc stabilité numérique attendue à backend fixé.

### 1.3 Couche per-unit parallélisée

Fichier : `src/hierarchicalcausalmodels/estimation/per_unit.py`

- `fit_per_unit_estimators(...)`
- `fit_regressors_per_unit(...)`
- `estimate_ate_confounder(...)`
- `fit_torch_batched_regressor_per_unit(...)`
- `estimate_ate_confounder_torch_batched(...)`
- `device_kwargs_for_workers(...)` (répartition CPU/GPU des workers)

### 1.4 Intégration dans l’estimateur causal HCM

Fichier : `src/hierarchicalcausalmodels/estimation/causal_estimators.py`

Parallélisation utilisée notamment pour :

- précalcul de variables `Q` (fit par unité),
- ajustement de blocs conditionnels par unité,
- évaluation par unité dans `ast_to_estimator` / `estimate_causal_effect`.

Paramètres clés :

- `n_jobs`
- `parallel_backend` (`threads` / `processes`)
- `estimator_backend` (`numpy` / `torch` / `numpyro`)

Règles spécifiques observées :

- `numpyro` : réduction forcée de parallélisme Python (`local_n_jobs=1`) pour éviter surcoût/instabilité (JAX/XLA fait déjà son parallélisme interne).
- si `numpyro` + `threads`, bascule de backend vers `processes` dans certains chemins.

## 2) Analyse expérimentale déjà faite (STAR)

Source : `examples/STAR/results/parallel_benchmark_results.json`

### 2.1 Batch économétrique STAR-like

Taille :

- `n_rows = 5725`
- `n_ols_specs = 24`
- `n_iv_specs = 10`

Temps mesurés :

- séquentiel :
  - OLS `12.3242 s`
  - IV `21.4296 s`
- threads (`n_jobs=4`) :
  - OLS `6.7054 s`
  - IV `14.8683 s`
- processes (`n_jobs=4`) :
  - OLS `0.4604 s`
  - IV `2.0457 s`

Stabilité coefficients (identiques à bruit numérique près) :

- OLS `small` (premier spec) : `6.508425193712797`
- IV `class size` (premier spec) : `-0.8182256773...`

Conclusion batch économétrique :

- gain net avec parallélisme,
- `processes` est de loin le plus performant sur ce workload.

### 2.2 Benchmark per-unit linéaire synthétique

Bloc `per_unit_linear` dans le même JSON :

- séquentiel :
  - fit `0.1443 s`
  - ate `0.1342 s`
- threads4 :
  - fit `0.2538 s`
  - ate `0.2624 s`
- proc4 :
  - fit `0.2277 s`
  - ate `0.2447 s`

ATE identique : `0.45036892041624194`.

Conclusion per-unit léger :

- la surcharge d’orchestration parallèle peut dominer,
- séquentiel peut être meilleur si chaque fit unitaire est très simple.

## 3) Démo et exposition API

Exemple dédié :

- `examples/new/parallel_estimators_demo.py`

La démo couvre :

- fit per-unit custom + agrégation ATE,
- fit per-unit sklearn parallèle,
- estimateur torch batched,
- allocation de devices workers.

Exports publics (package) :

- `src/hierarchicalcausalmodels/estimation/__init__.py`
- inclut `parallel_map`, `fit_many_ols`, `fit_many_2sls`, `estimate_ate_confounder`, `estimate_ate_confounder_torch_batched`, etc.

## 4) Interprétation architecture/performance

### 4.1 Pourquoi `processes` gagne en batch économétrique

- tâches indépendantes assez lourdes (24 OLS + 10 IV),
- bon amortissement du coût de démarrage/sérialisation,
- exploitation multi-coeur plus efficace que threads Python pour ce profil.

### 4.2 Pourquoi le per-unit léger peut perdre en parallèle

- chaque tâche est trop courte,
- overhead `joblib` > coût du calcul,
- benefit du parallélisme non amorti.

### 4.3 Implication pour `estimate_causal_effect`

Le gain dépend de la lourdeur réelle :

- graphe dense + termes coûteux + beaucoup d’unités : parallèle souvent utile,
- pipeline court / estimations triviales : séquentiel souvent optimal.

## 5) Règles d’usage recommandées

1. Batch économétrique (`fit_many_ols`, `fit_many_2sls`) :
   - commencer avec `n_jobs=4`, `parallel_backend="processes"`.
2. Per-unit simple :
   - démarrer séquentiel (`n_jobs=1`), puis profiler.
3. Backends probabilistes/variationnels (`numpyro`) :
   - éviter d’empiler parallélisme Python + parallélisme JAX.
4. Torch batched :
   - privilégier ce chemin pour gros batchs plutôt qu’un grand nombre de petits workers Python.
5. Toujours valider l’invariance numérique :
   - coefficients / ATE identiques (ou quasi) entre modes avant de retenir une config.

## 6) Risques et limites connus

- overhead de sérialisation sur objets lourds,
- contention mémoire si trop de workers,
- gains non monotones selon taille des tâches,
- certains chemins HCM peuvent être dominés par d’autres goulots (MC, fit densité, parsing formule), pas uniquement le dispatch parallèle.

## 7) Résumé exécutif

- La parallélisation est bien intégrée de façon transverse (économétrique, per-unit, HCM).
- Les benchmarks STAR confirment de gros gains en batch économétrique avec `processes`.
- Le per-unit léger ne bénéficie pas toujours du parallélisme.
- La stratégie optimale est adaptive : profiler par bloc, ne pas forcer un mode global.
