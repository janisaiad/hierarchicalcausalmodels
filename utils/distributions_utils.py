# functorization of the ppf, cdf and pdf functions for a given data to accelerate the computation

import numpy as np

def ppf_functor(data):
    return lambda q: np.percentile(data, q * 100)

def cdf_functor(data):
    return lambda x: np.searchsorted(data, x, side='right') / len(data)

def pdf_functor(data): # to modify
    return lambda x: np.histogram(data, bins='auto', density=True)[0]


def distribution_functor(d,random_samples,data,node,sizes):
    return ppf_functor(data), cdf_functor(data), pdf_functor(data)
