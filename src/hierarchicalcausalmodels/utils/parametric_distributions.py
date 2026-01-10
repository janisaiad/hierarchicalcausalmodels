from hierarchicalcausalmodels.utils.distributions_utils import ppf_functor, cdf_functor, pdf_functor
from scipy.stats import norm, uniform, expon, gamma, beta, lognorm, weibull_min, weibull_max, gumbel_r, gumbel_l, logistic, cauchy, chi2, f, t, triang, pareto, powerlaw

distribution_dict = { norm, uniform, expon, gamma, beta, lognorm, weibull_min, weibull_max, gumbel_r, gumbel_l, logistic, cauchy, chi2, f, t, triang, pareto, powerlaw }



class Distribution:
    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def ppf(self, q):
        raise NotImplementedError



class EmpiricalDistribution(Distribution):
    def __init__(self, data, distribution):
        self.data = data
        self.n = len(data)
        if distribution not in distribution_dict:
            raise ValueError("Distribution not supported")
        else:
            self.pdf = distribution.pdf
            self.cdf = distribution.cdf
            self.ppf = distribution.ppf