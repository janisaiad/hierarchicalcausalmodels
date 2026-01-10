# functorization of the ppf, cdf and pdf functions for a given data to accelerate the computation

import numpy as np


def create_subunit_distribution(data, node, unit_index, size):
    """
    Create a distribution from all subunit nodes for a given unit.

    :param data: The data dictionary containing all node values
    :param node: The base name of the node (e.g., 'b' for '_b3_k')
    :param unit_index: The index of the unit (e.g., 3 for '_b3_k')
    :param size: The number of subunits for this unit
    :return: A list of values for all subunits of this unit
    """
    subunit_values = [data[f'_{node}{unit_index}_{k}'] for k in range(size)]
    return subunit_values


def ppf_functor(data, node, unit_index, size):
    # Convert data to a numpy array if it's not already
    data_array = create_subunit_distribution(data, node, unit_index, size[unit_index]) # there was here a former indicator that it was working well
    x = np.random.random()
    return np.percentile(data_array, x * 100)


# ... existing code ...

def cdf_functor(data):
    # Convert data to a numpy array if it's not already
    data_array = np.array(list(data)) if isinstance(data, set) else np.array(data)
    return lambda x: np.searchsorted(data_array, x, side='right') / len(data_array)


def pdf_functor(data):
    # Convert data to a numpy array if it's not already
    #data_array = np.array(list(data)) if isinstance(data, set) else np.array(data)
    #hist, bin_edges = np.histogram(data_array, bins='auto', density=True)
    #return lambda x: np.interp(x, (bin_edges[:-1] + bin_edges[1:]) / 2, hist)
    return lambda x: 1


def distribution_functor(data, node, unit_index, sizes):
    return ppf_functor(data, node, unit_index, sizes)




def ppf_functor_unit(data):
    # Convert data to a numpy array if it's not already
    data_array = np.array(list(data)) if isinstance(data, set) else np.array(data)
    return lambda q: np.percentile(data_array, q * 100)