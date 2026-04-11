# STAR — benchmark familles non gaussiennes (numpy vs numpyro)

## Données et échelles

- unités (classes) : `28`
- quick : `True`
- NumPyro disponible : `True`

## Tableau comparatif

| Graphe | Scénario | Famille Y | Backend | Statut | E[do(1)] | ATE | Temps (s) | max(|ΔE_do|) vs numpy | erreur (extrait) |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `PC` | `gaussian` | `gaussian` | numpy | `ok` | 430.21429 | 0.00000 | 0.17 | |  |
| `DirectLiNGAM` | `gaussian` | `gaussian` | numpy | `error` |  |  | 0.79 | | X has 6 features, but LinearRegression is expecting 4 features as inpu |
| `PC` | `beta_unit` | `beta` | numpy | `ok` | 5.10514 | 0.00000 | 0.15 | |  |
| `PC` | `beta_unit` | `beta` | numpyro | `ok` | 5.10514 | 0.00000 | 0.02 | 0.00000 |  |
| `DirectLiNGAM` | `beta_unit` | `beta` | numpy | `error` |  |  | 0.77 | | X has 6 features, but LinearRegression is expecting 4 features as inpu |
| `DirectLiNGAM` | `beta_unit` | `beta` | numpyro | `error` |  |  | 0.80 |  | X has 6 features, but LinearRegression is expecting 4 features as inpu |
| `PC` | `gamma_positive_shift` | `gamma` | numpy | `ok` | 7.10341 | 0.00000 | 0.11 | |  |
| `PC` | `gamma_positive_shift` | `gamma` | numpyro | `ok` | 7.10341 | 0.00000 | 0.03 | 0.00000 |  |
| `DirectLiNGAM` | `gamma_positive_shift` | `gamma` | numpy | `error` |  |  | 0.86 | | X has 6 features, but LinearRegression is expecting 4 features as inpu |
| `DirectLiNGAM` | `gamma_positive_shift` | `gamma` | numpyro | `error` |  |  | 0.71 |  | X has 6 features, but LinearRegression is expecting 4 features as inpu |
| `PC` | `gaussian_mixture_raw` | `gaussian_mixture` | numpy | `ok` | 0.60761 | 0.00000 | 1.52 | |  |
| `PC` | `gaussian_mixture_raw` | `gaussian_mixture` | numpyro | `ok` | 0.60761 | 0.00000 | 1.41 | 0.00000 |  |
| `DirectLiNGAM` | `gaussian_mixture_raw` | `gaussian_mixture` | numpy | `error` |  |  | 1.72 | | X has 6 features, but LinearRegression is expecting 4 features as inpu |
| `DirectLiNGAM` | `gaussian_mixture_raw` | `gaussian_mixture` | numpyro | `error` |  |  | 1.96 |  | X has 6 features, but LinearRegression is expecting 4 features as inpu |

## Notes

- La colonne ATE pour `beta_unit` est sur le score de lecture linéairement renormalisé dans $(0,1)$, pas sur les points bruts du test.
- Pour `gamma_positive_shift`, l’ATE est sur $Y - \min(Y) + \epsilon$ (ordre de grandeur comparable au brut).
- `gaussian_mixture` reste le plus sensible à l’optimisation variationnelle ; comparer surtout l’ordre de grandeur avec numpy.
- Sur les graphes issus de la découverte causale packagée ici, la formule identifiée est souvent réduite à $P(Q^y)$ sans dépendance effective à $do(Q^a)$ : l’ATE est alors nul alors que $E[\mathrm{do}(1)]$ reste l’estimande principal à comparer entre backends.
