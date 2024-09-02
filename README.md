### README for Hierarchical Causal Models (HCM)

## Hierarchical Causal Models (HCM)

Hierarchical Causal Models (HCM) is a Python library for creating and analyzing hierarchical causal models. This library allows you to define nodes, edges, unit nodes, subunit nodes, and sizes, and provides functionality for sampling data and estimating effects.

## Installation

To install the HCM library, follow these steps:

### Prerequisites

- Python 3.11 or higher
- Poetry

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://code.ai-vidence.com/causalit/HCM.git
    cd hierarchicalcausalmodels
    ```

2. **Install Poetry**:
    If you don't have Poetry installed, you can install it using the following command:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
   or using pip:
    ```bash
    pip install poetry
    ```


3. **Install dependencies**:
    Use Poetry to install the project dependencies:
    ```bash
    poetry install
    ```

4. **Activate the virtual environment**:
    Activate the virtual environment created by Poetry:
    ```bash
    poetry shell
    ```

## Usage

Here is a basic example of how to use the HCM library:

```python
import numpy as np
from hierarchicalcausalmodels.models.HSCM import HSCM
from scipy.stats import norm

# Define the nodes, edges, unit nodes, subunit nodes, and sizes
nodes = ["a", "b", "c", "d", "e"]
edges = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("b", "d"), ("d", "e"), ("b", "e")]
unit_nodes = ["a", "b", "c", "e"]
subunit_nodes = ["d", "b"]
sizes = [3, 2]

# Define the additive functions
additive_functions = {
    "a": {},
    "b": {"a": lambda a: a * 0.1},
    "c": {"a": lambda a: a * 0.1, "b": lambda b: np.mean(np.array(list(b))) * 0.1},
    "d": {"b": lambda b: b * 0.1, "c": lambda c: c * 0.1},
    "e": {"d": lambda d: np.mean(np.array(list(d))) * 0.1, "b": lambda b: np.mean(np.array(list(b))) * 0.1}
}

# Define the randomness
randomness = {
    "a": lambda x: norm.ppf(x, 0, 1),
    "b": lambda x: norm.ppf(x, 0, 1),
    "c": lambda x: norm.ppf(x, 0, 1),
    "d": lambda x: norm.ppf(x, 0, 1),
    "e": lambda x: norm.ppf(x, 0, 1)
}

# Create an instance of the HSCM class
hscm = HSCM(nodes, edges, unit_nodes, subunit_nodes, sizes, node_functions={})
hscm.additive_model(additive_functions, randomness)

# Sample data
sampled_data = hscm.sample_data()

def print_sampled_data(sampled_data):
    for key in sampled_data:
        print(key, sampled_data[key])

# Print the sampled data
print_sampled_data(sampled_data)

# Print the graph
hscm.cgm.draw()
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.