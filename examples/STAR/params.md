# Paramètres retirés du PDF

Ce document centralise les paramètres techniques et hyperparamètres qui ont été retirés de `refs/report/report.tex`.

## 1) Paramètres d'estimation (`estimate_causal_effect`)

- Familles de distribution (exemple) :
  - `Q^{y|a}` : `gaussian`
  - `Q^a` : `gaussian`
- Intégration Monte Carlo :
  - `n_mc_samples = 1000`
- Reproductibilité :
  - `random_seed = 42`

## 2) Familles de `ConditionalDensityEstimator`

- Gaussienne : paramètres `mu, sigma` (liens identité/softplus)
- Beta : paramètres `alpha, beta` (lien softplus)
- Gamma : paramètres `k, theta` (lien softplus)
- Log-normale : paramètres `mu, sigma`
- Laplace : paramètres `mu, b`
- Bernoulli : paramètre `p` (lien sigmoïde)
- Poisson : paramètre `lambda` (lien softplus)
- Student-t : paramètres `nu, mu, sigma`
- Exponentielle : paramètre `lambda`
- Gaussienne inverse : paramètres `mu, lambda`
- Half-Cauchy : paramètre `gamma`
- Non-paramétrique : KDE / k-NN

## 3) Hyperparamètres réseau (`MLPRegressorPerUnit`)

- `hidden_layer_sizes = (64, 32)`
- `activation = "relu"`
- `max_iter = 500`
- `learning_rate_init = 1e-3`

## 4) Paramètres de génération HSCM (exemple confondeur)

- `U` : `mean = 0`, `std = 1`
- `A` : coefficient `U = 1.0`, `mean = 0`, `std = 0.5`
- `Y` : coefficients `A = 2.0`, `U = 1.0`, `mean = 0`, `std = 0.5`

## 5) Paramètre de famille dans le pipeline bout-en-bout (exemple)

- `distribution_families = {"Q^{y|a}": "gaussian"}`
