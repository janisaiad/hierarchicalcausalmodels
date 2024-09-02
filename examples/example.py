import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))



import numpy as np
from hierarchicalcausalmodels.models.HSCM import HSCM

# Define the node distributions.py
node_distributions = {
    "a": lambda d=dict: np.random.normal(0,1),
    "b": lambda a: sum(abs(k) for k in a.values()) + np.random.normal(0,1),
    "c": lambda b:  np.random.normal(0,1) + 2*sum(k for k in b.values()),
    "d": lambda c: sum(abs(k) for k in c.values()) + np.random.normal(0,1),
    "e": lambda d: sum(abs(k) for k in d.values()) + np.random.normal(0,1)
}


# Define the nodes, edges, unit nodes, subunit nodes, and sizes
nodes = ["a", "b", "c", "d", "e"]
edges = [("a", "b"),('a','c'), ("b", "c"), ("c", "d"),("b","d"), ("d", "e")]
unit_nodes = ["a", "b", "c"]
subunit_nodes = ["d", "e"]
sizes = [3, 2]

# Create an instance of the HSCM class
hscm = HSCM(nodes, edges, unit_nodes, subunit_nodes, sizes, node_distributions=node_distributions)
hscm.print_predecessors()
# Sample data
sampled_data = hscm.sample_data()

def print_sampled_data(sampled_data):
    for key in sampled_data:
        print(key, sampled_data[key])

# Print the sampled data
print_sampled_data(sampled_data)

# Print the graph
a= hscm.cgm.draw()
print(type(a))
print(hscm.cgm.draw())