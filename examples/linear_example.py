import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


from hierarchicalcausalmodels.models.HSCM.HSCM import HSCM
import numpy as np



coeffs = {
    "a": {"mean": 0, "std": 1},
    "b": {"a": 1.5, "mean": 0, "std": 1},
    "c": {"a": 0.5, "b": 2.0, "mean": 0, "std": 1},
    "d": {"b": 1.0, "c": 1.0, "mean": 0, "std": 1},
    "e": {"d": 3.0,"b" :1, "mean" : 0, "std": 1}
}

# Define the

# Define the nodes, edges, unit nodes, subunit nodes, and sizes
nodes = ["a", "b", "c", "d", "e"]
edges = [("a", "b"),('a','c'), ("b", "c"), ("c", "d"),("b","d"), ("d", "e"),("b", "e")]
unit_nodes = ["a", "c", "e"]
subunit_nodes = ["d","b"]
sizes = [100]*4



# Create an instance of the HSCM class
hscm = HSCM(nodes, edges, unit_nodes, subunit_nodes, sizes, node_functions={},data=[])
# hscm.print_predecessors()
hscm.linear_model(coeffs)
for edge in edges:
    hscm.set_aggregator(edge,lambda d:np.std(np.array(list(d))))
# Sample data
sampled_data = hscm.sample_data()

def print_sampled_data(sampled_data):
    for key in sampled_data:
        print(key, sampled_data[key])

# Print the sampled data
print_sampled_data(sampled_data)

# Print the graph
hscm.cgm.draw()
hscm.set_experimental_distributions_from_data()

hscm.plot_subunit_distributions()