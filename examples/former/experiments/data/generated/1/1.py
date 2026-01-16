import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','..')))

from models.HSCM.HSCM import HSCM # type: ignore
# ... rest of the code remains the same
import numpy as np
from scipy.stats import norm, expon, beta, gamma

# Define a large set of nodes
num_nodes = 50
nodes = [f"node_{i}" for i in range(num_nodes)]

# Define edges (create a complex graph structure)
edges = []
for i in range(num_nodes - 1):
    for j in range(i + 1, min(i + 5, num_nodes)):  # Each node connects to up to 4 subsequent nodes
        if np.random.random() < 0.7:  # 70% chance of connection
            edges.append((f"node_{i}", f"node_{j}"))

# Define unit nodes and subunit nodes
unit_nodes = [f"node_{i}" for i in range(0, num_nodes, 2)]  # Even nodes are unit nodes
subunit_nodes = [f"node_{i}" for i in range(1, num_nodes, 2)]  # Odd nodes are subunit nodes

# Define sizes (number of subunits for each unit)
num_units = 100
sizes = np.random.randint(50, 200, num_units)

# Define additive functions
additive_functions = {}
for node in nodes:
    additive_functions[node] = {}
    for parent, child in edges:
        if child == node:
            additive_functions[node][parent] = lambda x: np.random.choice([0.1, 0.5, 1.0, 2.0]) * x

# Define randomness (different distribution for each node)
distributions = [norm, expon, beta, gamma]
randomness = {}
for node in nodes:
    dist = np.random.choice(distributions)
    if dist == beta:
        randomness[node] = lambda x: dist.ppf(x, a=2, b=2)
    elif dist == gamma:
        randomness[node] = lambda x: dist.ppf(x, a=2)
    else:
        randomness[node] = lambda x: dist.ppf(x)

# Create an instance of the HSCM class
hscm = HSCM(nodes, edges, unit_nodes, subunit_nodes, sizes, node_functions={}, data=None)

# Set up the HSCM model
hscm.additive_model(additive_functions, randomness)

# Sample data
sampled_data = hscm.sample_data()

# Print some basic information about the generated data
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")
print(f"Number of unit nodes: {len(unit_nodes)}")
print(f"Number of subunit nodes: {len(subunit_nodes)}")
print(f"Number of units: {num_units}")

# Print a sample of the data
print("\nSample of generated data:")
for node in nodes[:5]:  # Print data for the first 5 nodes
    if node in unit_nodes:
        print(f"{node}: {sampled_data[node + '0']}")  # Print data for the first unit
    else:
        print(f"{node}: {sampled_data[node + '0_0']}")  # Print data for the first subunit of the first unit

# Optionally, save the data to a file
import pickle
with open('hierarchical_causal_data.pkl', 'wb') as f:
    pickle.dump(sampled_data, f)

print("\nData saved to 'hierarchical_causal_data.pkl'")

# Visualize the graph structure
hscm.cgm.draw()