import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))



import numpy as np
from hierarchicalcausalmodels.models.HSCM.HSCM import HSCM

from scipy.stats import norm

norm_cdf = norm.cdf

coeffs = {
    "a": {"mean": 0, "std": 1},
    "b": {"a": 1.5, "mean": 0, "std": 1},
    "c": {"a": 0.5, "b": 2.0, "mean": 0, "std": 1},
    "d": {"b": 1.0, "c": 1.0, "mean": 0, "std": 1},
    "e": {"d": 3.0,'b':1, "mean": 0, "std": 1}
}

# Define the

# Define the nodes, edges, unit nodes, subunit nodes, and sizes
nodes = ["a", "b", "c", "d", "e"]
edges = [("a", "b"),('a','c'), ("b", "c"), ("c", "d"),("b","d"), ("d", "e"),("b", "e")]
unit_nodes = ["a", "b", "c", "e"]
subunit_nodes = ["d","b"]
sizes = [3, 2]



def generator(node,d,x):
    aggregate = 0
    for j in d.values():
        print('j',j)
        if type(j) == set:
            aggregate += np.mean(np.array(list(j)))
        else:
            aggregate += j
    mean = aggregate
    std = np.sqrt(aggregate)+1
    return norm_cdf(x, loc=mean, scale=std)




# Create an instance of the HSCM class
hscm = HSCM(nodes, edges, unit_nodes, subunit_nodes, sizes, node_functions={})
# hscm.print_predecessors()
hscm.set_distributions_from_generator(generator)
hscm.random_model()

# Sample data
sampled_data = hscm.sample_data()

def print_sampled_data(sampled_data):
    for key in sampled_data:
        print(key, sampled_data[key])

# Print the sampled data
print_sampled_data(sampled_data)

# Print the graph
hscm.cgm.draw()