# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulations
# This notebook reproduces the simulation results from Section 4 of "Hierarchical Causal Models", E. N. Weinstein and D. M. Blei, 2023.
#
# To run the notebook, make sure you are using Python 3 and have the below packages installed in your environment. The code was tested with
#
# -- python=3.12.0
#
# -- numpy=1.26.2
#
# -- matplotlib=3.8.0
#
# -- scipy=1.11.4
#
# -- pandas=2.1.4
#
#

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli, norm
from scipy.special import logit, expit
from scipy import integrate
import pandas as pd

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

# Whether and where to save figures.
save_figs = False
save_folder = 'saved_figs/'


# %% [markdown]
# # CONFOUNDER model

# %% [markdown]
# #### Model & Estimators

# %%
class ConfounderModel:
    def __init__(self, U_omega, A_alphas_U, A_betas_U, YA_alphas_Ua, YA_betas_Ua):
        """Initialization: true data generating parameters."""
        # Parameter for p(u)
        self.U_omega = U_omega
        # Parameters for p(q^a|u)
        self.A_alphas_U = A_alphas_U
        self.A_betas_U = A_betas_U
        # Parameters for p(q^{y|a} | u)
        self.YA_alphas_Ua = YA_alphas_Ua
        self.YA_betas_Ua = YA_betas_Ua
    
    def sample(self, M, N):
        """Generate data."""
        # Sample U.
        U = bernoulli(self.U_omega).rvs(size=N)
        # Sample q^a.
        A_mu = beta(self.A_alphas_U[U], self.A_betas_U[U]).rvs()
        # Sample q^{y|a}
        YA_mu = beta(self.YA_alphas_Ua[U], self.YA_betas_Ua[U]).rvs()
        # Sample A.
        A = bernoulli(A_mu[:, None] * np.ones(M)).rvs()
        # Sample Y.
        Y = bernoulli(YA_mu[:, 0, None] * (1 - A) + YA_mu[:, 1, None] * A).rvs()
        
        return A, Y
    
    def true_effect(self, a):
        """Compute the true effect, E_p[E_Q[Y]|do(q^a = delta_a)]."""
        return ((1 - self.U_omega) * beta(self.YA_alphas_Ua[0, a], self.YA_betas_Ua[0, a]).mean() +
                self.U_omega * beta(self.YA_alphas_Ua[1, a], self.YA_betas_Ua[1, a]).mean())
    
    def estimate_effect(self, A, Y, a):
        """Estimate the effect using the HCM identification formula."""
        match = A == a
        # Use one pseudocount (posterior mean for Beta(1, 1) prior)
        return np.mean((np.sum(Y * match, axis=1) + 1) / (np.sum(match, axis=1) + 2))
    
    def regression_estimate(self, A, Y, a):
        """Estimate the effect using regression on the within-unit averages."""
        Amn = np.mean(A, axis=1, keepdims=True)
        Ymn = np.mean(Y, axis=1)
        reg = LinearRegression().fit(Amn, Ymn)
        return reg.predict(np.array([[a]]))[0]


# %% [markdown]
# #### Run simulations

# %%
# Data generating parameters
U_omegas = [0., 0.2, 0.5]
A_alphas_U = np.array([0.5, 4])
A_betas_U = np.array([1., 1.])
YA_alphas_Ua = np.array([[0.5, 2], [1, 4]])
YA_betas_Ua = np.ones((2, 2))
nsamps = [10, 100, 500, 1000]
reps = 20
ylim = [0, 0.7]

# Iterate over confounding levels.
for ui, U_omega in enumerate(U_omegas):

    # Initialize the model.
    model = ConfounderModel(U_omega, A_alphas_U, A_betas_U, YA_alphas_Ua, YA_betas_Ua)

    # Store estimates
    estimated_ate = np.zeros((reps, len(nsamps)))
    regress_ate = np.zeros((reps, len(nsamps)))
    
    # Random seed.
    np.random.seed(0)
    
    # Iterate over repetitions
    for i in range(reps):
        # Draw samples for full dataset.
        A, Y = model.sample(max(nsamps), max(nsamps))

        # Iterate over data set sizes.
        for j, nsamp in enumerate(nsamps):
            # Collect subset of full dataset. 
            A_sub, Y_sub = A[:nsamp, :nsamp], Y[:nsamp, :nsamp]
            # Compute confounder model-based estimate.
            estimated_ate[i, j] = (model.estimate_effect(A_sub, Y_sub, 1) - 
                                   model.estimate_effect(A_sub, Y_sub, 0))
            # Compute regression estimate.
            regress_ate[i, j] = (model.regression_estimate(A_sub, Y_sub, 1) - 
                               model.regression_estimate(A_sub, Y_sub, 0))
    # Compute true effect.
    true_ate = model.true_effect(1) - model.true_effect(0)

    # Plot results.
    plt.figure(figsize=(6, 5))
    plt.plot(nsamps, np.ones_like(nsamps) * true_ate, 'k:', label='True')
    plt.errorbar(nsamps, np.mean(regress_ate, axis=0), yerr=np.std(regress_ate, axis=0), color='red', 
                 linewidth=0, label='Regression', marker='x', capsize=4, elinewidth=2)
    plt.errorbar(nsamps, np.mean(estimated_ate, axis=0), yerr=np.std(estimated_ate, axis=0), color='blue', 
                 linewidth=0, label=r'CONFOUNDER', marker='x', capsize=4, elinewidth=2)
    if ui == 0:
        plt.legend(fontsize=16)
    plt.plot(nsamps, np.mean(regress_ate, axis=0), linewidth=3, alpha=0.4, color='red')
    plt.plot(nsamps, np.mean(estimated_ate, axis=0), linewidth=3, alpha=0.4, color='blue')
    plt.xlabel(r'Data size (# of units and # of subunits)', fontsize=16)
    plt.ylabel('Estimated treatment effect', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(ylim)
    plt.xscale('log')
    plt.tight_layout()

    if save_figs:
        plt.savefig(save_folder + 'unobserved_confound_strength_{}.pdf'.format(U_omega))


# %% [markdown]
# # CONFOUNDER & INTERFERENCE model

# %% [markdown]
# #### Model & Estimators

# %%
def normal_expit(b, c, x):
    # Function for integrating expit against Gaussian base distribution.
    return expit(b + c * x) * norm.pdf(x)
    
class InterferenceModel:
    def __init__(self, A_beta, A_scale, Z_tau, Z_mn, YA_delta, YA_rho, YA_beta, YA_sigma):
        """Initialization: true data generating parameters."""
        # Parameters for p(q^a | u)
        self.A_beta = A_beta
        self.A_scale = A_scale
        # Parameters for p(z | q^a)
        self.Z_tau = Z_tau
        self.Z_mn = Z_mn
        # Parameters for p(q^{y|a} | z, u)
        self.YA_delta = YA_delta
        self.YA_rho = YA_rho
        self.YA_beta = YA_beta
        self.YA_sigma = YA_sigma
    
    def sample(self, M, N):
        """Generate data."""
        # Sample U.
        U = norm.rvs(size=N)
        # Sample q^a.
        logit_A_rho = norm(self.A_beta * U, self.A_scale).rvs()
        A_rho = expit(logit_A_rho)
        # Sample A.
        A = bernoulli(A_rho[:, None] * np.ones(M)).rvs()
        # Sample Z.
        # ** delete: Z_mn = self.sigma(self.logit_est(np.mean(A, axis=1)))
        Z_mn = expit(self.Z_tau*(logit(np.mean(A, axis=1)) - self.Z_mn))
        Z = bernoulli(Z_mn).rvs()
        # Sample q^{y|a}
        logit_YA_rho = norm(np.array([0., self.YA_delta])[None, :] +
                            self.YA_rho * (2 * Z - 1)[:, None] +
                            self.YA_beta * U[:, None], 
                            self.YA_sigma).rvs()
        YA_rho = expit(logit_YA_rho)
        # Sample Y.
        Y = bernoulli(YA_rho[:, 0, None] * (1 - A) + YA_rho[:, 1, None] * A).rvs()
        
        return A, Z, Y
    
    def true_effect(self, qA):
        """Compute the true effect, E_p[E_Q[Y]|do(q^a = qA)]."""
        # Total effect.
        effect = 0
        # Iterate over a values.
        for a in range(2):
            if a == 1:
                a_term = qA
            else:
                a_term = 1 - qA
            # Iterate over z values.
            for z in range(2):
                pz = expit(self.Z_tau*(logit(qA) - self.Z_mn))
                if z == 1:
                    z_term = pz
                else:
                    z_term = 1 - pz

                # Compute the expectation with numerical integration.
                integ = integrate.quad(lambda u: normal_expit(self.YA_delta * a + self.YA_rho * (2 * z - 1), 
                                                              np.sqrt(self.YA_beta**2 + self.YA_sigma**2), u), -5, 5)

                # Add term to total effect.
                effect += a_term * z_term * integ[0]
            
        return effect
    
    def estimate_Q(self, A, Y):
        """Estimate the Q variables for each unit.""" 
        N, M = A.shape
        # Estimate q^a. We use pseudocounts for regularization, so our estimate is equivalent to
        # the posterior mean of a Beta-Bernoulli model with a Beta(1, 1) prior.
        A_mn_est = (np.sum(A, axis=1) + 1)/(M+2)
        # Estimate q^{y|a} 
        YA_mn_est = np.stack([(np.sum(Y * (A == 0), axis=1) + 1) / (np.sum(A == 0, axis=1) + 2),
                               (np.sum(Y * (A == 1), axis=1) + 1) / (np.sum(A == 1, axis=1) + 2)],
                              axis=1)
        return A_mn_est, YA_mn_est
    
    def estimate_effect(self, A, Z, Y, qAs, n_samps=100):
        """Estimate the effect using the HCM identification formula."""
        N, M = A.shape
        # Estimate subunit distribution.
        A_mn_est, YA_mn_est = self.estimate_Q(A, Y)
        
        # Learn regression from Q^A to Z
        lreg_z = LogisticRegression() 
        X1 = logit(A_mn_est)[:, None]
        lreg_z.fit(X1, Z)
        
        # Estimate Z probability under interventions.
        qA_transf = logit(qAs)[:, None]
        PZ_qAs = lreg_z.predict_proba(qA_transf)[:, 1]
        
        # Learn regression from Q^A and Z to Q^Y|A
        lreg_qya = [[None, None], [None, None]]
        lreg_qya_std = [[None, None], [None, None]]
        for a in range(2):
            for z in range(2):
                lreg_qya[a][z] = LinearRegression()
                logit_YA_mn_est = logit(YA_mn_est[Z == z][:, a])
                # Fit linear model.
                lreg_qya[a][z].fit(X1[Z == z], logit_YA_mn_est)
                # Estimate noise standard deviation from residual.
                lreg_qya_std[a][z] = np.sqrt(np.mean((lreg_qya[a][z].predict(X1[Z == z]) 
                                                      - logit_YA_mn_est)**2))
        
        # Compute effect estimate.
        effects = np.zeros(2)
        # Iterate over interventions.
        for i in range(2):
            eEl_QYA = np.zeros(2)
            # Iterate over z values.
            for z in range(2):
                # Compute z probability.
                if z == 0:
                    PZz_qA = 1 - PZ_qAs[i]
                elif z == 1:
                    PZz_qA = PZ_qAs[i]
                # Iterate over a values.
                for a in range(2):
                    # Sample from distribution over q^{y|a}.
                    qya_mu_logit = lreg_qya[a][z].predict(X1)
                    qya_mu_samples = expit(norm(qya_mu_logit[:, None] * np.ones((1, n_samps)), 
                                                lreg_qya_std[a][z]).rvs())
                    # Compute expected value of q^{y|a}.
                    E_qya = np.mean(qya_mu_samples, axis=1)
                    eEl_QYA[a] += np.mean(E_qya)*PZz_qA

            # Compute expected value of Y.
            effects[i] = eEl_QYA[1] * qAs[i] + eEl_QYA[0] * (1 - qAs[i])

        # Compute difference in treatment effects.
        return effects[1] - effects[0]

    def no_interfere_estimate(self, A, Z, Y, qAs):
        """Estimate the effect based on the CONFOUNDER graph (i.e. ignoring interference).""" 
        effects = np.zeros(2)
        for a in range(2):
            match = A == a
            effects[a] = np.mean((np.sum(Y * match, axis=1) + 1) / (np.sum(match, axis=1) + 2))
        return effects[1] * (qAs[1] - qAs[0]) + effects[0] * (qAs[0] - qAs[1])

    def regress_estimate(self, A, Y, qAs):
        """Estimate the effect using linear regression on aggregated data."""
        Amn = np.mean(A, axis=1, keepdims=True)
        Ymn = np.mean(Y, axis=1)
        reg = LinearRegression().fit(Amn, Ymn)
        return reg.predict(np.array([[qAs[1]]]))[0] - reg.predict(np.array([[qAs[0]]]))[0]


# %% [markdown]
# #### Run simulations

# %%
A_beta = 0.5
A_scale = 1.
Z_tau = 2. 
Z_mn = 0.4
YA_delta = 0.5
YA_rhos = [0, 0.5, 1.5]
YA_beta = 0.5
YA_sigma = 0.1
qAs = np.array([0.25, 0.75])
nsamps = [20, 100, 500, 1000]
reps = 20
ylim = [0., 0.6]

for rhoi, YA_rho in enumerate(YA_rhos):

    model = InterferenceModel(A_beta, A_scale, Z_tau, Z_mn, YA_delta, YA_rho, YA_beta, YA_sigma)
    
    estimated_effect = np.zeros((reps, len(nsamps)))
    no_interfere_effect = np.zeros((reps, len(nsamps)))
    regress_effect = np.zeros((reps, len(nsamps)))
    np.random.seed(1)
    for i in range(reps):
        A, Z, Y = model.sample(max(nsamps), max(nsamps))
        for j, nsamp in enumerate(nsamps):
            Z_sub, A_sub, Y_sub = Z[:nsamp], A[:nsamp, :nsamp], Y[:nsamp, :nsamp]
            estimated_effect[i, j] = model.estimate_effect(A_sub, Z_sub, Y_sub, qAs)
            no_interfere_effect[i, j] = model.no_interfere_estimate(A_sub, Z_sub, Y_sub, qAs)
            regress_effect[i, j] = model.regress_estimate(A_sub, Y_sub, qAs)
    true_effect = model.true_effect(qAs[1]) - model.true_effect(qAs[0])
    
    plt.figure(figsize=(6, 5))
    plt.plot(nsamps, np.ones_like(nsamps) * true_effect, 'k:', label='True')
    plt.errorbar(nsamps, np.mean(regress_effect, axis=0), yerr=np.std(regress_effect, axis=0),
             color='r', linewidth=0, label='Regression', marker='x', capsize=4, elinewidth=2)
    plt.errorbar(nsamps, np.mean(no_interfere_effect, axis=0), yerr=np.std(no_interfere_effect, axis=0),
             color='g', linewidth=0, label='CONFOUNDER', marker='x', capsize=4, elinewidth=2)
    plt.errorbar(nsamps, np.mean(estimated_effect, axis=0), yerr=np.std(estimated_effect, axis=0),
                 color='b', linewidth=0, label='CONFOUNDER & INTERFERENCE', marker='x', capsize=4, elinewidth=2)
    if rhoi == 0:
        plt.legend(fontsize=14)
    plt.plot(nsamps, np.mean(regress_effect, axis=0), linewidth=3, alpha=0.4, color='red')
    plt.plot(nsamps, np.mean(no_interfere_effect, axis=0), linewidth=3, alpha=0.4, color='green')
    plt.plot(nsamps, np.mean(estimated_effect, axis=0), linewidth=3, alpha=0.4, color='blue')
    plt.xscale('log')
    plt.xlabel('Data size (# of units and # of subunits)', fontsize=16)
    plt.ylabel('Estimated treatment effect', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(ylim)
    plt.tight_layout()

    if save_figs:
        plt.savefig(save_folder + 'interfere_converge_istrength_' + ('%.2f' % YA_rho) + '.pdf')


# %% [markdown]
# # INSTRUMENT model

# %% [markdown]
# #### Model & Estimators

# %%
class InstrumentModel:
    def __init__(self, U_rho, Z_alpha, Z_beta, AZ_alphas_Uz, AZ_betas_Uz, Y_rho_U, Y_eta, AZ_lim):
        """Initialization: true data generating parameters."""
        # Parameter for p(u).
        self.U_rho = U_rho
        # Parameters for p(q^z).
        self.Z_alpha = Z_alpha
        self.Z_beta = Z_beta
        # Parameters for p(q^{a|z}|u)
        self.AZ_alphas_Uz = AZ_alphas_Uz
        self.AZ_betas_Uz = AZ_betas_Uz
        self.AZ_lim = AZ_lim
        # Parameters for p(y|{a_j}, u)
        self.Y_rho_U = Y_rho_U
        self.Y_eta = Y_eta
        
    def sample(self, N, M):
        """Generate data."""
        # Sample U.
        U = bernoulli(self.U_rho).rvs(size=N)
        # Sample Q^z.
        Z_rho = beta(self.Z_alpha, self.Z_beta).rvs(size=N)
        # Sample Q^{a|z}
        AZ_rho = beta(self.AZ_alphas_Uz[U], self.AZ_betas_Uz[U]).rvs()
        AZ_rho[:, 0] = self.AZ_lim * AZ_rho[:, 0]
        AZ_rho[:, 1] = (1 - self.AZ_lim) + self.AZ_lim * AZ_rho[:, 1]
        # Sample Z.
        Z = bernoulli(Z_rho[:, None] * np.ones(M)).rvs()
        # Sample A.
        A = bernoulli(AZ_rho[:, 0, None] * (1 - Z) + AZ_rho[:, 1, None] * Z).rvs()
        # Sample Y.
        Y = bernoulli(self.Y_rho_U[U] * self.Y_eta + 
                      np.mean(A, axis=1) * (1 - self.Y_eta)).rvs()
        
        return Z, A, Y
    
    def true_effect(self, qA):
        """Compute the true effect."""
        return ((1 - self.U_rho) * self.Y_rho_U[0] * self.Y_eta +
                self.U_rho * self.Y_rho_U[1] * self.Y_eta +
                qA * (1 - self.Y_eta))
     
    
    def estimate_Q(self, Z, A):
        """Estimate the Q variables for each unit.""" 
        N, M = A.shape
        # Estimate q^a.
        A_mn_est = (np.sum(A, axis=1) + 1)/(M+2)
        # Estimate q^{a|z}.
        AZ0_mn_est = (np.sum(A * (Z == 0), axis=1) + 1) / (np.sum(Z == 0, axis=1) + 2)
        AZ1_mn_est = (np.sum(A * (Z == 1), axis=1) + 1) / (np.sum(Z == 1, axis=1) + 2)
        return A_mn_est, AZ0_mn_est, AZ1_mn_est
        
    def estimate_effect(self, Z, A, Y, qAs):
        """Estimate the effect using the HCM identification formula."""
        # Handle small data special cases.
        if np.all(Y == 1):
            return 1.
        elif np.all(Y == 0):
            return 0.
            
        # Estimate subunit distributions.
        N, M = Z.shape
        A_mn_est, AZ0_mn_est, AZ1_mn_est = self.estimate_Q(Z, A)
        
        # Train outcome regression.
        gpc = GaussianProcessClassifier()
        X = logit(np.stack([A_mn_est, AZ0_mn_est, AZ1_mn_est], axis=1))
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        gpc.fit(X, Y)
        
        # Compute backdoor correction.
        Xdo0 = logit(np.stack([np.ones(N) * qAs[0], AZ0_mn_est, AZ1_mn_est], axis=1))
        Xdo0 = scaler.transform(Xdo0)
        Y_est0 = gpc.predict_proba(Xdo0)
        Xdo1 = logit(np.stack([np.ones(N) * qAs[1], AZ0_mn_est, AZ1_mn_est], axis=1))
        Xdo1 = scaler.transform(Xdo1)
        Y_est1 = gpc.predict_proba(Xdo1)

        # Average over empirical distribution of q^{a|z}.
        return np.mean(Y_est1[:, 1]) - np.mean(Y_est0[:, 1])
    
    def naive_estimate(self, Z, A, Y, qAs):
        """Estimate the effect using linear regression (does not account for confounding)."""
        # Handle small data special cases.
        if np.all(Y == 1):
            return 1.
        elif np.all(Y == 0):
            return 0.
        
        # Estimate subunit treatment distribution.
        N, M = Z.shape
        A_mn_est, _, _ = self.estimate_Q(Z, A)
        
        # Train regression on treatment alone (naive approach).
        gpc = GaussianProcessClassifier()
        X = logit(A_mn_est[:, None])
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        gpc.fit(X, Y)
        
        # Form prediction.
        Y_est0 = gpc.predict_proba(scaler.transform(logit(np.array([[qAs[0]]]))))
        Y_est1 = gpc.predict_proba(scaler.transform(logit(np.array([[qAs[1]]])))) 
        return Y_est1[0, 1] - Y_est0[0, 1]
        

# %% [markdown]
# #### Run simulations
# Note: these take slightly longer to run (roughly a couple minutes).

# %%
def estimator_convergence(c_strength, reps=20):
    """Simulate data and evaluate estimators."""
    Z_alpha = 2.0
    Z_beta = 2.0
    AZ_alphas_Uz = np.array([[2, 2], [0.2, 0.2]])
    AZ_betas_Uz = np.array([[0.2, 0.2], [2, 2]])
    AZ_lim = 0.2
    Y_rho_U = np.array([0.9, 0.1])
    Y_eta = 0.5
    model = InstrumentModel(c_strength, Z_alpha, Z_beta, AZ_alphas_Uz, AZ_betas_Uz, Y_rho_U, Y_eta, AZ_lim)
    Z, A, Y = model.sample(1000, 1000)
    qAs = np.array([0.25, 0.75])
    
    nsamps = [20, 100, 500, 1000]
    estimated_effect = np.zeros((reps, len(nsamps)))
    regress_effect = np.zeros((reps, len(nsamps)))
    np.random.seed(1)
    true_effect = model.true_effect(qAs[1]) - model.true_effect(qAs[0])
    for i in range(reps):
        Z, A, Y = model.sample(max(nsamps), max(nsamps))
        for j, nsamp in enumerate(nsamps):
            Z_sub, A_sub, Y_sub = Z[:nsamp, :nsamp], A[:nsamp, :nsamp], Y[:nsamp]
            estimated_effect[i, j] = model.estimate_effect(Z_sub, A_sub, Y_sub, qAs)
            regress_effect[i, j] = model.naive_estimate(Z_sub, A_sub, Y_sub, qAs)
    return nsamps, true_effect, estimated_effect, regress_effect
    
def make_plot(c_strength, nsamps, true_effect, estimated_effect, regress_effect, save=False, ylim=[0, 1], 
              legend=True):
    """Plot results."""
    plt.figure(figsize=(6, 5))
    plt.plot(nsamps, np.ones_like(nsamps) * true_effect, 'k:', label='True')
    plt.errorbar(nsamps, np.mean(regress_effect, axis=0), yerr=np.std(regress_effect, axis=0),
             color='r', linewidth=0, label='Regression', marker='x', capsize=4, elinewidth=2)
    plt.errorbar(nsamps, np.mean(estimated_effect, axis=0), yerr=np.std(estimated_effect, axis=0),
                 color='b', linewidth=0, label='INSTRUMENT', marker='x', capsize=4, elinewidth=2)
    if legend:
        plt.legend(fontsize=16)
    plt.plot(nsamps, np.mean(regress_effect, axis=0), linewidth=3, alpha=0.4, color='red')
    plt.plot(nsamps, np.mean(estimated_effect, axis=0), linewidth=3, alpha=0.4, color='blue')
    plt.xscale('log')
    plt.xlabel('Data size (# of units and # of subunits)', fontsize=16)
    plt.ylabel('Estimated treatment effect', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(ylim)
    plt.tight_layout()
    if save_figs:
        plt.savefig(save_folder + 'converge_cstrength_' + ('%.2f' % c_strength) + '.pdf')


# %%
reps = 20
c_strength_low = 0.
nsamps_low, true_effect_low, estimated_effect_low, regress_effect_low = (
    estimator_convergence(c_strength_low, reps))
make_plot(c_strength_low, nsamps_low, true_effect_low, estimated_effect_low, regress_effect_low, 
          save=True, ylim=[0, 0.6], legend=True)
c_strength_med = 0.2
nsamps_med, true_effect_med, estimated_effect_med, regress_effect_med = (
    estimator_convergence(c_strength_med, reps))
make_plot(c_strength_med, nsamps_med, true_effect_med, estimated_effect_med, regress_effect_med, 
          save=True, ylim=[0, 0.6], legend=False)
c_strength_high = 0.5
nsamps_high, true_effect_high, estimated_effect_high, regress_effect_high = (
    estimator_convergence(c_strength_high, reps))
make_plot(c_strength_high, nsamps_high, true_effect_high, estimated_effect_high, regress_effect_high, 
          save=True, ylim=[0, 0.6], legend=False)

# %% [markdown]
# # GENERALIZED CAUSAL MODEL
#
# A unified framework that handles arbitrary causal graph structures, not limited to specific patterns like confounding, interference, or instruments.
#

# %% [markdown]
# #### Model Definition

# %%

from abc import ABC, abstractmethod
import itertools
import networkx as nx
from networkx.algorithms.d_separation import is_d_separator
from typing import Dict, List, Callable

class CausalNode(ABC):
    """Abstract base class for nodes in a causal graph."""

    def __init__(self, name: str):
        self.name = name
        self.parents = []
        self.children = []

    @abstractmethod
    def sample(self, parent_values: Dict[str, np.ndarray], target_shape) -> np.ndarray:
        """Generate samples given parent values."""
        pass

    def add_parent(self, parent):
        """Register a parent node."""
        if parent not in self.parents:
            self.parents.append(parent)

    def add_child(self, child):
        """Register a child node."""
        if child not in self.children:
            self.children.append(child)

    def _call_func(self, func: Callable, parent_values: Dict[str, np.ndarray], target_shape):
        try:
            return func(parent_values, target_shape)
        except TypeError:
            return func(parent_values)

    def _normalize_shape(self, target_shape):
        if target_shape is None:
            return None
        if isinstance(target_shape, int):
            return (target_shape,)
        if isinstance(target_shape, (tuple, list)):
            return tuple(target_shape)
        return None

    def _as_samples(self, values, target_shape):
        shape = self._normalize_shape(target_shape)
        arr = np.asarray(values)
        if shape is None:
            return arr
        if arr.shape == shape:
            return arr
        if arr.shape == () or arr.size == 1:
            return np.full(shape, arr.item())
        try:
            return np.broadcast_to(arr, shape)
        except ValueError as exc:
            raise ValueError('Function outputs must broadcast to target shape') from exc
class BernoulliNode(CausalNode):
    """Bernoulli node: generates binary outcomes based on parent probabilities."""

    def __init__(self, name: str, prob_func: Callable):
        """
        Args:
            name: Node identifier
            prob_func: Function that maps parent values to success probability
        """
        super().__init__(name)
        self.prob_func = prob_func

    def sample(self, parent_values: Dict[str, np.ndarray], target_shape) -> np.ndarray:
        probs = self._call_func(self.prob_func, parent_values, target_shape)
        probs = self._as_samples(probs, target_shape)
        probs = np.clip(probs, 0, 1)
        return bernoulli(probs).rvs()


class BetaNode(CausalNode):
    """Beta node: generates continuous outcomes on [0,1] using Beta distribution."""

    def __init__(self, name: str, alpha_func: Callable, beta_func: Callable):
        """
        Args:
            name: Node identifier
            alpha_func: Function that maps parent values to alpha parameter
            beta_func: Function that maps parent values to beta parameter
        """
        super().__init__(name)
        self.alpha_func = alpha_func
        self.beta_func = beta_func

    def sample(self, parent_values: Dict[str, np.ndarray], target_shape) -> np.ndarray:
        alphas = self._call_func(self.alpha_func, parent_values, target_shape)
        betas = self._call_func(self.beta_func, parent_values, target_shape)
        alphas = self._as_samples(alphas, target_shape)
        betas = self._as_samples(betas, target_shape)
        alphas = np.clip(alphas, 0.01, np.inf)
        betas = np.clip(betas, 0.01, np.inf)
        return beta(alphas, betas).rvs()


class GaussianNode(CausalNode):
    """Gaussian node: generates continuous outcomes from Normal distribution."""

    def __init__(self, name: str, mean_func: Callable, std_func: Callable = None):
        """
        Args:
            name: Node identifier
            mean_func: Function that maps parent values to mean
            std_func: Function that maps parent values to standard deviation (default: constant 1.0)
        """
        super().__init__(name)
        self.mean_func = mean_func
        self.std_func = std_func if std_func is not None else (lambda p, s: np.ones(self._normalize_shape(s) or (1,)))

    def sample(self, parent_values: Dict[str, np.ndarray], target_shape) -> np.ndarray:
        means = self._call_func(self.mean_func, parent_values, target_shape)
        stds = self._call_func(self.std_func, parent_values, target_shape)
        means = self._as_samples(means, target_shape)
        stds = self._as_samples(stds, target_shape)
        stds = np.clip(stds, 0.01, np.inf)
        return norm(means, stds).rvs()


class GeneralizedCausalModel:
    """
    A flexible framework for specifying and simulating any causal graph structure.

    Supports arbitrary DAGs with mixed variable types (binary, continuous, etc.)
    and flexible functional relationships between variables.
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.graph = nx.DiGraph()

    def add_node(self, node: CausalNode):
        """Add a node to the causal model."""
        self.nodes[node.name] = node
        self.graph.add_node(node.name)

    def add_edge(self, parent_name: str, child_name: str):
        """Add a causal edge from parent to child."""
        if parent_name not in self.nodes or child_name not in self.nodes:
            raise ValueError("Both parent and child nodes must be added first")

        parent = self.nodes[parent_name]
        child = self.nodes[child_name]
        child.add_parent(parent)
        parent.add_child(child)
        self.graph.add_edge(parent_name, child_name)

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order for correct sampling."""
        return list(nx.topological_sort(self.graph))

    def validate_dag(self) -> bool:
        """Verify that the graph is a valid DAG (no cycles)."""
        return nx.is_directed_acyclic_graph(self.graph)

    def sample(self, n_samples) -> Dict[str, np.ndarray]:
        """
        Generate data from the causal model.

        Args:
            n_samples: Number of samples to generate (int/tuple) or dict of node->shape

        Returns:
            Dictionary mapping node names to generated samples
        """
        if not self.validate_dag():
            raise ValueError("Graph contains cycles. Must be a valid DAG.")

        samples = {}

        # Sample nodes in topological order
        for node_name in self.topological_sort():
            node = self.nodes[node_name]

            # Collect parent values
            parent_values = {}
            for parent in node.parents:
                parent_values[parent.name] = samples[parent.name]

            if isinstance(n_samples, dict):
                target_shape = n_samples.get(node_name)
                if target_shape is None and parent_values:
                    target_shape = np.asarray(next(iter(parent_values.values()))).shape
            else:
                target_shape = n_samples
                if parent_values:
                    target_shape = np.asarray(next(iter(parent_values.values()))).shape

            if target_shape is None:
                raise ValueError("Target shape could not be resolved for node")

            # Sample this node
            samples[node_name] = node.sample(parent_values, target_shape)

        return samples
    def _flatten_sample(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        return arr.ravel()

    def get_causal_dag_repr(self) -> str:
        """Return a string representation of the causal DAG."""
        edges = list(self.graph.edges())
        return f"Nodes: {list(self.nodes.keys())} Edges: {edges}"

    def _backdoor_graph(self, treatment: str) -> nx.DiGraph:
        g_bd = self.graph.copy()
        g_bd.remove_edges_from([(treatment, child) for child in self.graph.successors(treatment)])
        return g_bd

    def _candidate_adjustment_nodes(self, treatment: str, outcome: str) -> List[str]:
        nodes = set(self.nodes.keys()) - {treatment, outcome}
        descendants = nx.descendants(self.graph, treatment)
        return sorted(nodes - descendants)

    def find_backdoor_adjustment_sets(self,
                                      treatment: str,
                                      outcome: str,
                                      max_set_size: int = None) -> List[List[str]]:
        """
        Return minimal adjustment sets that satisfy the backdoor criterion.
        """
        if not self.validate_dag():
            raise ValueError("Graph contains cycles. Must be a valid DAG.")

        candidates = self._candidate_adjustment_nodes(treatment, outcome)
        g_bd = self._backdoor_graph(treatment)
        max_size = len(candidates) if max_set_size is None else min(max_set_size, len(candidates))

        for r in range(max_size + 1):
            found = []
            for combo in itertools.combinations(candidates, r):
                if is_d_separator(g_bd, treatment, outcome, set(combo)):
                    found.append(list(combo))
            if found:
                return found
        return []

    def find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> List[str] | None:
        sets = self.find_backdoor_adjustment_sets(treatment, outcome)
        return sets[0] if sets else None

    def estimate_causal_effect(self,
                               samples: Dict[str, np.ndarray],
                               treatment: str,
                               outcome: str,
                               adjustment_set: List[str] = None,
                               method: str = 'regression') -> float:
        """
        Estimate causal effect based on the graph structure and data.

        Uses the backdoor criterion to identify valid adjustment sets and
        applies appropriate identification strategies.

        Args:
            samples: Dictionary of variable samples
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            adjustment_set: Variables to adjust for (if None, auto-identifies)
            method: Estimation method ('regression', 'stratification', 'matching')

        Returns:
            Estimated causal effect
        """
        if adjustment_set is None:
            adjustment_set = self.find_backdoor_adjustment_set(treatment, outcome)
            if adjustment_set is None:
                raise ValueError("No valid backdoor adjustment set found")

        if method == 'regression':
            return self._estimate_via_regression(samples, treatment, outcome, adjustment_set)
        if method == 'stratification':
            return self._estimate_via_stratification(samples, treatment, outcome, adjustment_set)
        raise ValueError(f"Unknown estimation method: {method}")

    def _find_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Backward-compatible wrapper for earlier examples."""
        return self.find_backdoor_adjustment_set(treatment, outcome)

    def _estimate_via_regression(self,
                                 samples: Dict[str, np.ndarray],
                                 treatment: str,
                                 outcome: str,
                                 adjustment_set: List[str]) -> float:
        """Estimate causal effect via regression adjustment."""
        X = np.column_stack([self._flatten_sample(samples[treatment])] + [self._flatten_sample(samples[var]) for var in adjustment_set])
        y = self._flatten_sample(samples[outcome])

        reg = LinearRegression().fit(X, y)

        # Effect = difference in predicted outcome when treatment changes from 0 to 1
        X_0 = np.column_stack([np.zeros(X.shape[0])] + [samples[var] for var in adjustment_set])
        X_1 = np.column_stack([np.ones(X.shape[0])] + [samples[var] for var in adjustment_set])

        effect = np.mean(reg.predict(X_1) - reg.predict(X_0))
        return effect

    def _estimate_via_stratification(self,
                                     samples: Dict[str, np.ndarray],
                                     treatment: str,
                                     outcome: str,
                                     adjustment_set: List[str]) -> float:
        """Estimate causal effect via stratification (Mantel-Haenszel style)."""
        if not adjustment_set:
            treated = samples[outcome][samples[treatment] == 1]
            control = samples[outcome][samples[treatment] == 0]
            return np.mean(treated) - np.mean(control)

        data = {treatment: self._flatten_sample(samples[treatment]), outcome: self._flatten_sample(samples[outcome])}
        for var in adjustment_set:
            data[var] = self._flatten_sample(samples[var])
        df = pd.DataFrame(data)

        strata_effects = []
        strata_weights = []

        for _, group in df.groupby(adjustment_set):
            treated = group[group[treatment] == 1][outcome]
            control = group[group[treatment] == 0][outcome]
            if len(treated) > 0 and len(control) > 0:
                strata_effects.append(treated.mean() - control.mean())
                strata_weights.append(len(control))

        if strata_effects:
            return np.average(strata_effects, weights=strata_weights)
        return 0.0

    def estimate_effect_hcm(self, samples: Dict[str, np.ndarray], treatment: str, outcome: str, a: int) -> float:
        """Estimate effect using the HCM identification formula for binary treatment."""
        A = np.asarray(samples[treatment])
        Y = np.asarray(samples[outcome])
        if A.ndim != 2 or Y.ndim != 2:
            raise ValueError('HCM estimator expects 2D arrays (units x replications)')
        if A.shape != Y.shape:
            raise ValueError('Treatment/outcome shapes must match for HCM estimator')
        match = A == a
        return np.mean((np.sum(Y * match, axis=1) + 1) / (np.sum(match, axis=1) + 2))

    def estimate_ate_hcm(self, samples: Dict[str, np.ndarray], treatment: str = 'A', outcome: str = 'Y') -> float:
        """Estimate ATE using the HCM identification formula."""
        return (self.estimate_effect_hcm(samples, treatment, outcome, 1) -
                self.estimate_effect_hcm(samples, treatment, outcome, 0))



# %% [markdown]
# #### Examples: Recreating Previous Models with Generalized Framework

# %%
# Example 1: Recreate the Confounder Model as a Generalized Causal Model
print("=" * 60)
print("EXAMPLE 1: Confounder Model (U -> A, U -> Y)")
print("=" * 60)

def build_confounder_gcm(U_omega, A_alphas_U, A_betas_U, YA_alphas_Ua, YA_betas_Ua):
    """Build the confounder model using the generalized framework."""
    gcm = GeneralizedCausalModel()

    # Node U: Latent confounder (Bernoulli)
    U_node = BernoulliNode(
        "U",
        lambda parents: U_omega
    )
    gcm.add_node(U_node)

    # Node A_mu: group-level treatment propensity
    def A_alpha_func(parents):
        U_vals = parents.get("U")
        return A_alphas_U[U_vals.astype(int)]

    def A_beta_func(parents):
        U_vals = parents.get("U")
        return A_betas_U[U_vals.astype(int)]

    A_mu_node = BetaNode("A_mu", A_alpha_func, A_beta_func)
    gcm.add_node(A_mu_node)
    gcm.add_edge("U", "A_mu")

    # Node A: individual treatment draws
    A_node = BernoulliNode("A", lambda parents: parents.get("A_mu"))
    gcm.add_node(A_node)
    gcm.add_edge("A_mu", "A")

    # Nodes Y_mu0, Y_mu1: group-level outcome propensities for each treatment
    def Y0_alpha_func(parents):
        U_vals = parents.get("U")
        return YA_alphas_Ua[U_vals.astype(int), 0]

    def Y0_beta_func(parents):
        U_vals = parents.get("U")
        return YA_betas_Ua[U_vals.astype(int), 0]

    def Y1_alpha_func(parents):
        U_vals = parents.get("U")
        return YA_alphas_Ua[U_vals.astype(int), 1]

    def Y1_beta_func(parents):
        U_vals = parents.get("U")
        return YA_betas_Ua[U_vals.astype(int), 1]

    Y0_node = BetaNode("Y_mu0", Y0_alpha_func, Y0_beta_func)
    Y1_node = BetaNode("Y_mu1", Y1_alpha_func, Y1_beta_func)
    gcm.add_node(Y0_node)
    gcm.add_node(Y1_node)
    gcm.add_edge("U", "Y_mu0")
    gcm.add_edge("U", "Y_mu1")

    # Node Y: outcome drawn using A and group-level propensities
    def Y_prob(parents):
        A_vals = parents.get("A")
        Y0 = parents.get("Y_mu0")
        Y1 = parents.get("Y_mu1")
        return Y0 * (1 - A_vals) + Y1 * A_vals

    Y_node = BernoulliNode("Y", Y_prob)
    gcm.add_node(Y_node)
    gcm.add_edge("A", "Y")
    gcm.add_edge("Y_mu0", "Y")
    gcm.add_edge("Y_mu1", "Y")

    return gcm


# Instantiate with same parameters as before
U_omega = 0.5
A_alphas_U = np.array([0.5, 4])
A_betas_U = np.array([1., 1.])
YA_alphas_Ua = np.array([[0.5, 2], [1, 4]])
YA_betas_Ua = np.ones((2, 2))

gcm_confounder = build_confounder_gcm(U_omega, A_alphas_U, A_betas_U, YA_alphas_Ua, YA_betas_Ua)
print("\nConfounder Model DAG:")
print(gcm_confounder.get_causal_dag_repr())

# Sample data
np.random.seed(42)
samples_conf = gcm_confounder.sample(500)
print(f"\nSample shapes: A={samples_conf['A'].shape}, Y={samples_conf['Y'].shape}")
print(f"Mean A: {np.mean(samples_conf['A']):.3f}, Mean Y: {np.mean(samples_conf['Y']):.3f}")


# %%
# Compare ConfounderModel vs GeneralizedCausalModel estimates
np.random.seed(123)
N = 500
M = 500
conf_model = ConfounderModel(U_omega, A_alphas_U, A_betas_U, YA_alphas_Ua, YA_betas_Ua)
A_conf, Y_conf = conf_model.sample(M, N)
ate_conf = conf_model.estimate_effect(A_conf, Y_conf, 1) - conf_model.estimate_effect(A_conf, Y_conf, 0)

np.random.seed(124)
samples_gcm = gcm_confounder.sample({'U': N, 'A': (N, M), 'Y': (N, M)})
ate_gcm = gcm_confounder.estimate_ate_hcm(samples_gcm, 'A', 'Y')

true_ate = conf_model.true_effect(1) - conf_model.true_effect(0)
print(f'True ATE (ConfounderModel): {true_ate:.4f}')
print(f'ConfounderModel ATE: {ate_conf:.4f}')
print(f'GCM ATE: {ate_gcm:.4f}')


# %%
# Example 2: Complex Causal Graph with Mediation
print("\n" + "=" * 60)
print("EXAMPLE 2: Mediation Model (U -> A, A -> M -> Y, U -> Y)")
print("=" * 60)

def build_mediation_gcm():
    """
    Build a mediation model: 
    - U is a confounder of A and Y
    - A affects Y both directly and through mediator M
    - M is affected by A and U
    """
    gcm = GeneralizedCausalModel()
    
    # Confounder U
    U_node = BernoulliNode("U", lambda p: 0.4)
    gcm.add_node(U_node)
    
    # Treatment A (affected by U)
    def A_prob(parents):
        U_vals = parents.get("U", np.array([0]))
        return 0.3 + 0.3 * U_vals
    
    A_node = BernoulliNode("A", A_prob)
    gcm.add_node(A_node)
    gcm.add_edge("U", "A")
    
    # Mediator M (affected by A and U)
    def M_prob(parents):
        A_vals = parents.get("A", np.array([0]))
        U_vals = parents.get("U", np.array([0]))
        return 0.2 + 0.4 * A_vals + 0.3 * U_vals
    
    M_node = BernoulliNode("M", M_prob)
    gcm.add_node(M_node)
    gcm.add_edge("A", "M")
    gcm.add_edge("U", "M")
    
    # Outcome Y (affected by A, M, and U)
    def Y_prob(parents):
        A_vals = parents.get("A", np.array([0]))
        M_vals = parents.get("M", np.array([0]))
        U_vals = parents.get("U", np.array([0]))
        return 0.3 + 0.35 * A_vals + 0.3 * M_vals + 0.2 * U_vals
    
    Y_node = BernoulliNode("Y", Y_prob)
    gcm.add_node(Y_node)
    gcm.add_edge("A", "Y")
    gcm.add_edge("M", "Y")
    gcm.add_edge("U", "Y")
    
    return gcm

gcm_mediation = build_mediation_gcm()
print("\nMediation Model DAG:")
print(gcm_mediation.get_causal_dag_repr())

# Sample data
np.random.seed(42)
samples_med = gcm_mediation.sample(500)
print(f"\nSample shapes: A={samples_med['A'].shape}, M={samples_med['M'].shape}, Y={samples_med['Y'].shape}")

# Estimate total effect of A on Y
adjustment_set_total = gcm_mediation._find_adjustment_set("A", "Y")
print(f"\nAdjustment set for total effect (A->Y): {adjustment_set_total}")
total_effect = gcm_mediation.estimate_causal_effect(samples_med, "A", "Y", 
                                                     adjustment_set=adjustment_set_total)
print(f"Total effect of A on Y: {total_effect:.4f}")

# Estimate direct effect (excluding mediation path)
adjustment_set_direct = gcm_mediation._find_adjustment_set("A", "Y")
# For direct effect, add M to adjustment set to block mediation path
direct_effect = gcm_mediation.estimate_causal_effect(samples_med, "A", "Y", 
                                                      adjustment_set=adjustment_set_direct + ["M"])
print(f"Direct effect of A on Y (controlling for M): {direct_effect:.4f}")
print(f"Indirect effect (mediated through M): {total_effect - direct_effect:.4f}")


# %%
# Example 3: Selection Bias / Collider Graph
print("\n" + "=" * 60)
print("EXAMPLE 3: Collider Graph (A -> C <- Y, no direct A->Y)")
print("=" * 60)

def build_collider_gcm():
    """
    Build a collider model:
    - A and Y are independent
    - Both affect C (the collider)
    - Conditioning on C creates spurious correlation
    """
    gcm = GeneralizedCausalModel()
    
    # Treatment A (exogenous)
    A_node = BernoulliNode("A", lambda p: 0.5)
    gcm.add_node(A_node)
    
    # Outcome Y (exogenous, independent of A)
    Y_node = BernoulliNode("Y", lambda p: 0.5)
    gcm.add_node(Y_node)
    
    # Collider C (affected by both A and Y)
    def C_prob(parents):
        A_vals = parents.get("A", np.array([0]))
        Y_vals = parents.get("Y", np.array([0]))
        return 0.1 + 0.4 * A_vals + 0.4 * Y_vals
    
    C_node = BernoulliNode("C", C_prob)
    gcm.add_node(C_node)
    gcm.add_edge("A", "C")
    gcm.add_edge("Y", "C")
    
    return gcm

gcm_collider = build_collider_gcm()
print("\nCollider Model DAG:")
print(gcm_collider.get_causal_dag_repr())

# Sample data
np.random.seed(42)
samples_coll = gcm_collider.sample(1000)

# Check independence of A and Y in full data
corr_marginal = np.corrcoef(samples_coll['A'], samples_coll['Y'])[0, 1]
print(f"\nMarginal correlation between A and Y: {corr_marginal:.4f} (should be ~0)")

# Now condition on C=1 (selection bias)
mask_C1 = samples_coll['C'] == 1
if np.sum(mask_C1) > 10:
    corr_conditional = np.corrcoef(samples_coll['A'][mask_C1], samples_coll['Y'][mask_C1])[0, 1]
    print(f"Conditional correlation (given C=1): {corr_conditional:.4f} (spurious correlation!)")


# %%
# Example 4: M-Bias / Butterfly Graph
print("\n" + "=" * 60)
print("EXAMPLE 4: M-Bias Graph (U1 -> A -> C <- U2 -> Y)")
print("=" * 60)

def build_m_bias_gcm():
    """
    Build an M-bias (butterfly) graph:
    - U1 affects A and C
    - U2 affects C and Y
    - True path: A (exogenous) -> no direct effect -> Y (exogenous)
    - But conditioning on C opens a backdoor path
    """
    gcm = GeneralizedCausalModel()
    
    # Unmeasured confounder U1
    U1_node = BernoulliNode("U1", lambda p: 0.4)
    gcm.add_node(U1_node)
    
    # Unmeasured confounder U2
    U2_node = BernoulliNode("U2", lambda p: 0.3)
    gcm.add_node(U2_node)
    
    # Treatment A (affected by U1)
    def A_prob(parents):
        U1_vals = parents.get("U1", np.array([0]))
        return 0.2 + 0.5 * U1_vals
    
    A_node = BernoulliNode("A", A_prob)
    gcm.add_node(A_node)
    gcm.add_edge("U1", "A")
    
    # Collider C (affected by A, U1, U2)
    def C_prob(parents):
        A_vals = parents.get("A", np.array([0]))
        U1_vals = parents.get("U1", np.array([0]))
        U2_vals = parents.get("U2", np.array([0]))
        return 0.1 + 0.3 * A_vals + 0.3 * U1_vals + 0.35 * U2_vals
    
    C_node = BernoulliNode("C", C_prob)
    gcm.add_node(C_node)
    gcm.add_edge("A", "C")
    gcm.add_edge("U1", "C")
    gcm.add_edge("U2", "C")
    
    # Outcome Y (affected by U2 only)
    def Y_prob(parents):
        U2_vals = parents.get("U2", np.array([0]))
        return 0.3 + 0.4 * U2_vals
    
    Y_node = BernoulliNode("Y", Y_prob)
    gcm.add_node(Y_node)
    gcm.add_edge("U2", "Y")
    
    return gcm

gcm_mbias = build_m_bias_gcm()
print("\nM-Bias Model DAG:")
print(gcm_mbias.get_causal_dag_repr())

# Sample data
np.random.seed(42)
samples_mb = gcm_mbias.sample(1000)

# True effect should be 0 (A doesn't affect Y)
# Without adjusting for C
adjustment_set_unadjusted = gcm_mbias._find_adjustment_set("A", "Y")
print(f"\nAdjustment set without C: {adjustment_set_unadjusted}")
effect_unadjusted = gcm_mbias.estimate_causal_effect(samples_mb, "A", "Y", 
                                                      adjustment_set=adjustment_set_unadjusted)
print(f"Effect without adjusting for C: {effect_unadjusted:.4f} (correct, ~0)")

# With C included (opening backdoor)
effect_adjusted = gcm_mbias.estimate_causal_effect(samples_mb, "A", "Y", 
                                                    adjustment_set=["C"])
print(f"Effect when adjusting for C: {effect_adjusted:.4f} (spurious!)")


# %% [markdown]
# #### Comparison: Specific vs. Generalized Models

# %%
print("\n" + "=" * 70)
print("SUMMARY: Generalized vs. Specific Models")
print("=" * 70)

summary = """
ADVANTAGES OF GENERALIZED CAUSAL MODEL:

1. FLEXIBILITY
   - Handles ANY DAG structure, not limited to specific patterns
   - Supports mixed variable types (Bernoulli, Beta, Gaussian, custom)
   - Custom functional relationships between variables

2. GRAPH-BASED REASONING
   - Automatically identifies valid adjustment sets via backdoor criterion
   - Validates the graph is a valid DAG
   - Enables graphical analysis of causal structure

3. UNIFIED FRAMEWORK
   - Single API for data generation and effect estimation
   - Consistent estimation procedures across different models
   - Easy comparison between different causal structures

4. EXTENSIBILITY
   - Add new node types by extending CausalNode
   - Implement different estimation methods
   - Support continuous/hierarchical variables

RELATIONSHIP TO PREVIOUS MODELS:

- ConfounderModel: Special case of GCM with U -> A, U -> Y, A -> Y
- InterferenceModel: GCM with network effects through intermediate Z
- InstrumentModel: GCM with Z as instrument (A <- Z, U -> A, U -> Y)

KEY FEATURES DEMONSTRATED:

✓ Confounder graphs (confounding by unmeasured variables)
✓ Mediation graphs (direct and indirect effects)  
✓ Collider stratification (selection bias, M-bias)
✓ Instrument variables (handling endogeneity)
✓ Arbitrary graph structures

ESTIMATION STRATEGY:

1. Specify the DAG structure
2. Add nodes with their conditional distributions
3. Auto-identify valid adjustment sets
4. Apply regression or stratification
5. Validate against true causal effects
"""

print(summary)

