import numpy as np
from hierarchicalcausalmodels.utils.distributions_utils import ppf_functor, cdf_functor, pdf_functor
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