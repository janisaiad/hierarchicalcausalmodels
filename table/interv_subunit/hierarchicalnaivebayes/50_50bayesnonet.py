# %%

# %%
import pymc as pm
import numpy as np
import jax
import multiprocessing

#multiprocessing.set_start_method('spawn', force=True)
# %%
from scipy.stats import norm, gaussian_kde


def kl_divergence(p, q, bandwidth='scott'):
    """
    Calcule la divergence KL entre deux distributions empiriques représentées par des tableaux,
    en utilisant l'estimation de densité par noyau.

    :param p: Premier tableau de données
    :param q: Second tableau de données
    :param bandwidth: Méthode pour estimer la largeur de bande ('scott', 'silverman' ou un nombre)
    :return: Valeur de la divergence KL
    """
    # Assurez-vous que les tableaux ont la même taille
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]

    # Estimation de densité par noyau
    kde_p = gaussian_kde(p, bw_method=bandwidth)
    kde_q = gaussian_kde(q, bw_method=bandwidth)

    # Créez un espace d'échantillonnage
    x = np.linspace(min(np.min(p), np.min(q)), max(np.max(p), np.max(q)), 10000)

    # Estimez les densités
    p_density = kde_p(x)
    q_density = kde_q(x)

    # Ajoutez un petit epsilon pour éviter la division par zéro
    epsilon = 1e-10
    p_density += epsilon
    q_density += epsilon

    # Normalisez les densités
    p_density /= np.sum(p_density)
    q_density /= np.sum(q_density)

    # Calculez la divergence KL
    return np.sum(p_density * np.log(p_density / q_density))


# %%
import json


def load_data_from_json(file_path):
    """
    Load data from a JSON file and return it as a dictionary.

    :param file_path: Path to the JSON file
    :return: Dictionary containing the loaded data
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        print(f"Data successfully loaded from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


# %%
data = load_data_from_json('data/sampled_data.json')
# %%

graph = [('a', '_b'), ('a', 'c'), ('_b', 'c'), ('c', '_d'), ('_b', '_d'), ('_d', 'e'), ('c', 'e')]
n_schools = 50
n_students = 50

unit_vars = ['a', 'c', 'e']
subunit_vars = ['b', 'd']
sizes = [n_students] * n_schools
# %%

# %%

# Hypothèses : Tu as des données disponibles sous forme de matrices
# A, C, E : (100,)
# B, D : (100, 50)
"""
A = np.random.normal(0, 1, 100)
C = np.random.normal(0, 1, 100)
E = np.random.normal(0, 1, 100)

B_observed = np.random.normal(0, 1, (100, 50))
D_observed = np.random.normal(0, 1, (100, 50))
"""

a = np.array([data[f'a{i}'] for i in range(n_schools)])
c = np.array([data[f'c{i}'] for i in range(n_schools)])
e = np.array([data[f'e{i}'] for i in range(n_schools)])

# For b and d, we need to create 2D arrays
b = np.array([[data[f'_b{i}_{j}'] for j in range(n_students)] for i in range(n_schools)])
d = np.array([[data[f'_d{i}_{j}'] for j in range(n_students)] for i in range(n_schools)])

# %%
print(a)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Variables', fontsize=16)

# Plot distribution of a
sns.histplot(a, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of a')
axes[0, 0].set_xlabel('Value')

# Plot distribution of c
sns.histplot(c, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of c')
axes[0, 1].set_xlabel('Value')

# Plot distribution of e
sns.histplot(e, kde=True, ax=axes[0, 2])
axes[0, 2].set_title('Distribution of e')
axes[0, 2].set_xlabel('Value')

# Plot distribution of first b array
sns.histplot(b[0], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of first b array')
axes[1, 0].set_xlabel('Value')

# Plot distribution of first d array
sns.histplot(d[0], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of first d array')
axes[1, 1].set_xlabel('Value')

# Remove the unused subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# %%
print("Shape of a:", a.shape)
print("Shape of b:", b.shape)
print("Shape of c:", c.shape)
print("Shape of d:", d.shape)
print("Shape of e:", e.shape)

# %% md

# %%
with pm.Model() as model:
    # Niveau des écoles
    mu_A = pm.Normal('mu_A', mu=0, sigma=1)
    sigma_A = pm.HalfNormal('sigma_A', sigma=1)
    A_j = pm.Normal('A_j', mu=mu_A, sigma=sigma_A, observed=a)

    mu_C = pm.Normal('mu_C', mu=0, sigma=1)
    sigma_C = pm.HalfNormal('sigma_C', sigma=1)
    C_j = pm.Normal('C_j', mu=mu_C, sigma=sigma_C, observed=c)

    mu_E = pm.Normal('mu_E', mu=0, sigma=1)
    sigma_E = pm.HalfNormal('sigma_E', sigma=1)
    E_j = pm.Normal('E_j', mu=mu_E, sigma=sigma_E, observed=e)

    # B est maintenant une variable observée suivant une loi normale standard
    B_ij = pm.Normal('B_ij', mu=0, sigma=1, observed=b)

    # Modèle pour le poids des élèves (D) en fonction des autres variables
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta_A = pm.Normal('beta_A', mu=0, sigma=1)
    beta_C = pm.Normal('beta_C', mu=0, sigma=1)
    beta_E = pm.Normal('beta_E', mu=0, sigma=1)
    beta_B = pm.Normal('beta_B', mu=0, sigma=1)
    sigma_D = pm.HalfNormal('sigma_D', sigma=1)

    mu_D = alpha + beta_A * A_j[:, None] + beta_C * C_j[:, None] + beta_E * E_j[:, None] + beta_B * B_ij
    D_ij = pm.Normal('D_ij', mu=mu_D, sigma=sigma_D, observed=d)

    # Inférence
    trace = pm.sample(1000, return_inferencedata=True)

# Résumé des résultats
print(pm.summary(trace, var_names=['mu_A', 'mu_C', 'mu_E', 'alpha', 'beta_A', 'beta_C', 'beta_E', 'beta_B']))
# %%
generated_data_cond = {}

# a
mu_A_posterior = trace.posterior['mu_A'].mean(dim=('chain', 'draw')).values
sigma_A_posterior = trace.posterior['sigma_A'].mean(dim=('chain', 'draw')).values
new_a = np.random.normal(mu_A_posterior, sigma_A_posterior, size=n_schools)
generated_data_cond['a'] = new_a

# c
mu_C_posterior = trace.posterior['mu_C'].mean(dim=('chain', 'draw')).values
sigma_C_posterior = trace.posterior['sigma_C'].mean(dim=('chain', 'draw')).values
new_c = np.random.normal(mu_C_posterior, sigma_C_posterior, size=n_schools)
generated_data_cond['c'] = new_c

# e
mu_E_posterior = trace.posterior['mu_E'].mean(dim=('chain', 'draw')).values
sigma_E_posterior = trace.posterior['sigma_E'].mean(dim=('chain', 'draw')).values
new_e = np.random.normal(mu_E_posterior, sigma_E_posterior, size=n_schools)
generated_data_cond['e'] = new_e

# Pour B
generated_data_cond['b'] = np.random.normal(0, 1, size=(n_schools, n_students))

# Pour D, nous devons recalculer mu_D en utilisant les paramètres estimés
alpha = trace.posterior['alpha'].mean(dim=('chain', 'draw')).values
beta_A = trace.posterior['beta_A'].mean(dim=('chain', 'draw')).values
beta_C = trace.posterior['beta_C'].mean(dim=('chain', 'draw')).values
beta_E = trace.posterior['beta_E'].mean(dim=('chain', 'draw')).values
beta_B = trace.posterior['beta_B'].mean(dim=('chain', 'draw')).values
sigma_D = trace.posterior['sigma_D'].mean(dim=('chain', 'draw')).values

# Calcul de mu_D
mu_D = (alpha +
        beta_A * new_a[:, np.newaxis] +
        beta_C * new_c[:, np.newaxis] +
        beta_E * new_e[:, np.newaxis] +
        beta_B * generated_data_cond['b'])

generated_data_cond['d'] = np.random.normal(mu_D, sigma_D)

# Affichage des formes des données générées
for var in ['a', 'b', 'c', 'd', 'e']:
    print(f"Shape of generated {var}: {generated_data_cond[var].shape}")

print("Generated data:", generated_data_cond)
# %%
# Calcul des divergences KL
kl_divs = {}

for var in ['a', 'b', 'c', 'd', 'e']:
    original = eval(var)  # Les données originales
    generated = generated_data_cond[var]  # Les données générées

    # Pour b et d, nous devons aplatir les arrays 2D
    if var in ['b', 'd']:
        original = original.flatten()
        generated = generated.flatten()

    kl_divs[var] = kl_divergence(original, generated)

# Affichage des résultats
for var, kl_div in kl_divs.items():
    print(f"KL divergence pour {var}: {kl_div}")
# %% md

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Variables to plot
variables = ['a', 'c', 'e']

# Set up the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparison of Original and Generated Distributions', fontsize=16)

for i, var in enumerate(variables):
    ax = axes[i]

    # Original data
    original = eval(var)

    # Generated data
    generated = generated_data_cond[var]

    # KDE plot
    sns.kdeplot(original, ax=ax, label='Original', color='blue')
    sns.kdeplot(generated, ax=ax, label='Generated', color='red')

    # Histogram
    ax.hist(original, bins=30, alpha=0.3, color='blue', density=True)
    ax.hist(generated, bins=30, alpha=0.3, color='red', density=True)

    # Fit normal distribution
    mu_orig, std_orig = stats.norm.fit(original)
    mu_gen, std_gen = stats.norm.fit(generated)

    x = np.linspace(min(original.min(), generated.min()),
                    max(original.max(), generated.max()), 100)

    ax.plot(x, stats.norm.pdf(x, mu_orig, std_orig),
            'b--', linewidth=2, label='Original Gaussian Fit')
    ax.plot(x, stats.norm.pdf(x, mu_gen, std_gen),
            'r--', linewidth=2, label='Generated Gaussian Fit')

    ax.set_title(f'Distribution of {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.show()

# %%
# Variables to plot
variables_bd = ['b', 'd']

# Set up the plot
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
fig.suptitle('Comparison of Original and Generated Distributions for b and d', fontsize=16)

for i, var in enumerate(variables_bd):
    for j in range(50):
        ax = axes[5 * i + j // 10, j % 10]

        # Original data
        original = eval(f'{var}')[j]

        # Generated data
        generated = generated_data_cond[f'{var}'][j]

        # KDE plot
        sns.kdeplot(original, ax=ax, label='Original', color='blue')
        sns.kdeplot(generated, ax=ax, label='Generated', color='red')

        # Histogram
        ax.hist(original, bins=30, alpha=0.3, color='blue', density=True)
        ax.hist(generated, bins=30, alpha=0.3, color='red', density=True)

        # Fit normal distribution
        mu_orig, std_orig = stats.norm.fit(original)
        mu_gen, std_gen = stats.norm.fit(generated)

        x = np.linspace(min(original.min(), generated.min()),
                        max(original.max(), generated.max()), 100)

        ax.plot(x, stats.norm.pdf(x, mu_orig, std_orig),
                'b--', linewidth=1, label='Original Gaussian Fit')
        ax.plot(x, stats.norm.pdf(x, mu_gen, std_gen),
                'r--', linewidth=1, label='Generated Gaussian Fit')

        ax.set_title(f'{var}{j}', fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.legend(fontsize=6)

plt.show()

# %%
import json
import os


# Function to save sampled data to a JSON file
def save_sampled_data(data, filename):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)

    # Write new data to file
    print(f"Saving data to {filename}")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print("Data saved successfully")


# %%
def experiment(n):
    all_data = []
    all_kl_divs = []
    for k in range(n):
        data = load_data_from_json(f'data/sampled_intervention_{k}.json')

        a = np.array([data[f'a{i}'] for i in range(n_schools)])
        c = np.array([data[f'c{i}'] for i in range(n_schools)])
        e = np.array([data[f'e{i}'] for i in range(n_schools)])

        # For b and d, we need to create 2D arrays
        b = np.array([[data[f'_b{i}_{j}'] for j in range(n_students)] for i in range(n_schools)])
        d = np.array([[data[f'_d{i}_{j}'] for j in range(n_students)] for i in range(n_schools)])

        with pm.Model() as model:
            # Niveau des écoles
            mu_A = pm.Normal('mu_A', mu=0, sigma=1)
            sigma_A = pm.HalfNormal('sigma_A', sigma=1)
            A_j = pm.Normal('A_j', mu=mu_A, sigma=sigma_A, observed=a)

            mu_C = pm.Normal('mu_C', mu=0, sigma=1)
            sigma_C = pm.HalfNormal('sigma_C', sigma=1)
            C_j = pm.Normal('C_j', mu=mu_C, sigma=sigma_C, observed=c)

            mu_E = pm.Normal('mu_E', mu=0, sigma=1)
            sigma_E = pm.HalfNormal('sigma_E', sigma=1)
            E_j = pm.Normal('E_j', mu=mu_E, sigma=sigma_E, observed=e)

            # Niveau des élèves
            mu_B = pm.Normal('mu_B', mu=0, sigma=1)
            sigma_B = pm.HalfNormal('sigma_B', sigma=1)
            B_ij = pm.Normal('B_ij', mu=mu_B, sigma=sigma_B, observed=b)

            # Modèle pour le poids des élèves (D) en fonction des autres variables
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta_A = pm.Normal('beta_A', mu=0, sigma=1)
            beta_C = pm.Normal('beta_C', mu=0, sigma=1)
            beta_E = pm.Normal('beta_E', mu=0, sigma=1)
            beta_B = pm.Normal('beta_B', mu=0, sigma=1)

            beta_D_A3 = pm.Normal('beta_D_A3', mu=1, sigma=1)
            beta_D_A2 = pm.Normal('beta_D_A2', mu=1, sigma=1)

            sigma_D = pm.HalfNormal('sigma_D', sigma=1)

            mu_D = alpha + beta_A * A_j[:, None] + beta_C * C_j[:, None] + beta_E * E_j[:,
                                                                                    None] + beta_B * B_ij + beta_D_A3 * A_j[
                                                                                                                        :,
                                                                                                                        None] ** 3 + beta_D_A2 * A_j[
                                                                                                                                                 :,
                                                                                                                                                 None] ** 2
            D_ij = pm.Normal('D_ij', mu=mu_D, sigma=sigma_D, observed=d)

            # Inférence
            trace = pm.sample(50, return_inferencedata=True)

        # Résumé des résultats
        print(pm.summary(trace,
                         var_names=['mu_A', 'mu_C', 'mu_E', 'mu_B', 'alpha', 'beta_A', 'beta_C', 'beta_E', 'beta_B','beta_D_A2','beta_D_A3']))

        generated_data_cond = {}

        # a
        mu_A_posterior = trace.posterior['mu_A'].mean(dim=('chain', 'draw')).values
        sigma_A_posterior = trace.posterior['sigma_A'].mean(dim=('chain', 'draw')).values
        new_a = np.random.normal(mu_A_posterior, sigma_A_posterior, size=n_schools)
        generated_data_cond['a'] = new_a
        np.save(f'data/generated_cond_a_{k}.npy', new_a)
        # c
        mu_C_posterior = trace.posterior['mu_C'].mean(dim=('chain', 'draw')).values
        sigma_C_posterior = trace.posterior['sigma_C'].mean(dim=('chain', 'draw')).values
        new_c = np.random.normal(mu_C_posterior, sigma_C_posterior, size=n_schools)
        generated_data_cond['c'] = new_c
        np.save(f'data/generated_cond_c_{k}.npy', new_c)
        # e
        mu_E_posterior = trace.posterior['mu_E'].mean(dim=('chain', 'draw')).values
        sigma_E_posterior = trace.posterior['sigma_E'].mean(dim=('chain', 'draw')).values
        new_e = np.random.normal(mu_E_posterior, sigma_E_posterior, size=n_schools)
        generated_data_cond['e'] = new_e
        np.save(f'data/generated_cond_e_{k}.npy', new_e)
        # Pour B
        generated_data_cond['b'] = np.random.normal(0, 1, size=(n_schools, n_students))

        np.save(f'data/generated_cond_b_{k}.npy', generated_data_cond['b'])

        # Pour D, nous devons recalculer mu_D en utilisant les paramètres estimés
        alpha = trace.posterior['alpha'].mean(dim=('chain', 'draw')).values
        beta_A = trace.posterior['beta_A'].mean(dim=('chain', 'draw')).values
        beta_C = trace.posterior['beta_C'].mean(dim=('chain', 'draw')).values
        beta_E = trace.posterior['beta_E'].mean(dim=('chain', 'draw')).values
        beta_B = trace.posterior['beta_B'].mean(dim=('chain', 'draw')).values
        sigma_D = trace.posterior['sigma_D'].mean(dim=('chain', 'draw')).values

        # Calcul de mu_D
        mu_D = (alpha +
                beta_A * new_a[:, np.newaxis] +
                beta_C * new_c[:, np.newaxis] +
                beta_E * new_e[:, np.newaxis] +
                beta_B * generated_data_cond['b'])

        generated_data_cond['d'] = np.random.normal(mu_D, sigma_D)

        np.save(f'data/generated_cond_d_{k}.npy', generated_data_cond['d'])


        kl_divs = {}

        for var in ['a', 'b', 'c', 'd', 'e']:
            original = eval(var)  # Les données originales
            generated = generated_data_cond[var]  # Les données générées

            # Pour b et d, nous devons aplatir les arrays 2D
            if var in ['b', 'd']:
                original = original.flatten()
                generated = generated.flatten()  # be careful with is, it's all kl divs ..., not a kl div in 2d or a kl div for each

            kl_divs[var] = kl_divergence(original, generated)
        all_data.append((data, generated_data_cond))

        all_kl_divs.append(kl_divs)

    return all_kl_divs, all_data


# %%
kl_divs, all_data = experiment(100)
# %%

# %%
print(len(all_data))
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for plotting
variables = list(kl_divs[0].keys())  # Assuming kldivs is a list of dictionaries

# Create subplots for each variable
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, var in enumerate(variables):
    # Extract KL divergence values for the current variable
    kl_values = [kldiv[var] for kldiv in kl_divs]

    # Create a KDE plot for the current variable
    sns.kdeplot(kl_values, ax=axes[i], shade=True, color='skyblue')

    # Customize the subplot
    axes[i].set_title(f'KL Divergence Distribution for Variable {var}')
    axes[i].set_xlabel('KL Divergence')
    axes[i].set_ylabel('Density')

# Remove the empty subplot
fig.delaxes(axes[5])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# %%
import numpy as np

# Calculate mean and standard deviation for each variable
for var in variables:
    kl_values = [kldiv[var] for kldiv in kl_divs]
    mean = np.mean(kl_values)
    std = np.std(kl_values)
    print(f"Variable {var}:")
    print(f"  Mean KL Divergence: {mean:.4f}")
    print(f"  Standard Deviation: {std:.4f}")
    print()
