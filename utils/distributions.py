import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hierarchicalcausalmodels.utils.distributions_utils import ppf_functor, cdf_functor, pdf_functor # type: ignore

from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.special import kl_div

class Distribution:
    def __init__(self, distribution):
        self.distribution = distribution
        self.ppf = distribution.ppf
        self.cdf = distribution.cdf
        self.pdf = distribution.pdf
        pass
    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def ppf(self, q):
        raise NotImplementedError
    def kl_divergence(self, other, num_samples=10000):
        x = np.linspace(self.ppf(0.001), self.ppf(0.999), num_samples)
        p = self.pdf(x)
        q = other.pdf(x)
        return np.sum(kl_div(p, q))

    def wasserstein_distance(self, other, num_samples=10000):
        x = np.linspace(0, 1, num_samples)
        return wasserstein_distance(self.ppf(x), other.ppf(x))



class EmpiricalDistribution(Distribution):
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.ppf_functor = ppf_functor(data)
        self.cdf_functor = cdf_functor(data)
        self.pdf_functor = pdf_functor(data)
    def pdf(self, x):
        # Kernel Density Estimation can be used for a smoother PDF
        return self.pdf_functor(x)

    def cdf(self, x):
        return self.cdf_functor(x)

    def ppf(self, q):
        return self.ppf_functor(q)