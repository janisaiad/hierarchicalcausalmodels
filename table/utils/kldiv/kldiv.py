from scipy.stats import entropy
import numpy as np
def calculate_kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
