import pandas as pd
import numpy as np
from hierarchicalcausalmodels.models.HSCM.HSCM import HSCM  # type: ignore
import matplotlib.pyplot as plt
from scipy.stats import norm, expon

# Define the HSCM model structure
nodes = {"a", "b", "c", "d", "e"}
edges = {("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("b", "d"), ("d", "e"), ("c", "e")}
unit_nodes = {"a", "c", "e"}
subunit_nodes = {"d", "b"}
sizes = [2] * 2  # You can adjust this based on your data

# Initialize the HSCM model
hscm = HSCM(nodes, edges, unit_nodes, subunit_nodes, sizes, node_functions={}, data=None)

coeffs = {
    "a": {'mean': 0, 'std': 1},
    "b": {"a": 3, 'mean': 0, 'std': 1},
    "c": {"a": 2, "b": 3, 'mean': 0, 'std': 1},
    "d": {"b": 1.5, "c": 0.4, 'mean': 0, 'std': 1},
    "e": {"c": 2, "d": 0.1, 'mean': 0, 'std': 1}
}

# Set up the HSCM model
hscm.linear_model(coeffs)
print()

# Sample data from the model
sampled_data = hscm.sample_data()
print(sampled_data)

# Plot the sampled data
hscm.plot_data()

# Set distributions from the loaded data
hscm.set_distribution_from_data() # -> to modify

# Perform additional analysis or modify the model as needed
# For example, you can change the graph structure or random functions here

# Re-sample data after modifications
new_sampled_data = hscm.sample_data()
