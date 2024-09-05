# functorization of the ppf, cdf and pdf functions for a given data to accelerate the computation

import numpy as np

def ppf_functor(data):
    # Convert data to a numpy array if it's not already
    data_array = np.array(list(data)) if isinstance(data, set) else np.array(data)
    x = np.random.random()
    return np.percentile(data_array, x * 100)

# ... existing code ...

def cdf_functor(data):
    # Convert data to a numpy array if it's not already
    data_array = np.array(list(data)) if isinstance(data, set) else np.array(data)
    return lambda x: np.searchsorted(data_array, x, side='right') / len(data_array)

def pdf_functor(data):
    # Convert data to a numpy array if it's not already
    data_array = np.array(list(data)) if isinstance(data, set) else np.array(data)
    hist, bin_edges = np.histogram(data_array, bins='auto', density=True)
    return lambda x: np.interp(x, (bin_edges[:-1] + bin_edges[1:]) / 2, hist)


def distribution_functor(d,data,node,sizes):
    return ppf_functor(data), cdf_functor(data), pdf_functor(data)
