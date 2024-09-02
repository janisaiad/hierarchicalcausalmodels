import numpy as np
from hierarchicalcausalmodels.utils.distributions import EmpiricalDistribution
from hierarchicalcausalmodels.utils.parsing_utils import source, extract_number1
import pandas as pd

def additive_functor(d, node, additive_functions, is_unit, randomness):
    # for additive model + random gaussian noise
    s = 0
    for key in d.keys():  # for each predecessor in the parent sample
        if key[0] == '_':  # if the previous is a subunit node

            if is_unit:  # d[key] is a dictionary of the subunit values, we use the aggregator that already given in general
                s += additive_functions[node][source(key)](d[
                                                               key])  # if the previous is a subunit node and the node is a unit node (we take the mean for a linear model)
            else:  # if the previous is a subunit node and the node is a subunit node
                s += additive_functions[node][source(key)](
                    d[key])  # we are with the same person in the same subunit, transitions are simple functions R -> R
        else:
            s += additive_functions[node][source(key)](d[key])
    if is_unit:
        return s + randomness[node](np.random.uniform())
    return s + randomness[node[1:]](np.random.uniform())  # we add the final noise


def random_functor(d, node, distributions, aggregator):  # d is a dictionary of predecessors values
    return distributions[node](d, np.random.uniform())  # random generator according to the distribution


# ad-hoc models for linear and logistic regression
def linear_functor(d, node, coeffs, is_unit, aggregator):
    # for linear model + random gaussain noise
    s = 0
    for key in d.keys():
        if key[0] == '_':  # if the previous is a subunit node

            if is_unit:
                s += coeffs[node][source(key)] * aggregator[node][source(key)](d[key])  # if the previous is a subunit node and the node is a unit node, we take the mean for a linear model
            else:  # if the previous is a subunit node and the node is a subunit node
                s += coeffs[node][source(key)] * d[key]  # we are with the same person in the same subunit
        else:
            s += coeffs[node][source(key)] * d[key]
    return s + np.random.normal(coeffs[node]['mean'], coeffs[node]["std"])


def logit_functor(d, node, coeffs, is_unit, aggregator):
    s = 0
    for key in d.keys():
        if key[0] == '_':  # if the previous is a subunit node

            if is_unit:
                s += coeffs[node][source(key)] * aggregator[node][key[:-1]](d[
                                                                             key])  # if the previous is a subunit node and the node is a unit node, we take the mean for a linear model
            else:  # if the previous is a subunit node and the node is a subunit node
                s += coeffs[node][source(key)] * d[key]  # we are with the same person in the same subunit
        else:
            s += coeffs[node][source(key)] * d[key]
    return 1 / (1 + np.exp(s + np.random.normal(coeffs[node]['mean'], coeffs[node]["std"])))


# to modify
def distribution_functor(d, random_sample, data, node, sizes):
    # this will produce a family of distributions parametrized by the unit (3 schools = 3 distributions)
    emp_dists = dict()
    for i in range(sizes[node]):
        emp_dists[i] = EmpiricalDistribution({data[node + str(i) + '_' + str(j)] for j in len(sizes[i])})
    # now we have to recover in which unit we are

    if d.keys[0][0] == '_':  # ie we miraculously got a subunit node ... we have to recover the unit number
        return lambda d, random_sample: emp_dists[extract_number1(d.keys[0])].ppf(random_sample)

    return lambda d, random_sample: emp_dists[extract_number1(d.keys[0])].ppf(random_sample)


def is_empty(d):
    return not bool(d)


def cleaner(df: pd.DataFrame) -> dict:
    result = {}
    for column in df.columns:
        non_null_values = df[column].dropna().tolist()
        result[column] = dict(non_null_values)
    return result


def extract_distributions_from_data(data,nodes,unit_nodes,sizes):
    distributions = {}
    for node in nodes:
        if node in unit_nodes:
            node_distribution = EmpiricalDistribution({data[node+str(i)] for i in range(len(sizes))})
            distributions[node] = node_distribution
        else:
            distributions[node] = dict() # source(node) = 'a' or '_b3'
            for i in range(len(sizes)):
                subunit_node_distribution = EmpiricalDistribution({data[node+str(i)+'_' + str(j)] for j in range(sizes[i])})
                distributions[node][node+str(i)] = subunit_node_distribution
    return distributions # so distributions is {'a' : EmpDist(..), '_b' : { '_b1' : EmpDist .., ..}, .. }


