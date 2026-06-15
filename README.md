## Hierarchical Causal Models (HCM)

Hierarchical Causal Models (HCM) is a Python library for creating and analyzing hierarchical causal models. This library allows you to define nodes, edges, unit nodes, subunit nodes, and sizes, and provides functionality for sampling data and estimating effects.

## Installation

To install the HCM library, follow these steps:

### Prerequisites   

- Python 3.13 or higher
- `uv` (fast Python package installer)

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/janisaiad/hierarchicalcausalmodels
    cd hierarchicalcausalmodels
    ```

2. **Install uv**:
    If you don't have `uv` installed, you can install it using the official installer:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    Or using pip:
    ```bash
    pip install uv
    ```

3. **Create and activate virtual environment**:
    Create a virtual environment and activate it:
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4. **Install the package in editable mode**:
    Install the package and its dependencies:
    ```bash
    uv pip install -e .
    ```

Alternatively, you can use `uv` to run commands directly without activating the virtual environment:
```bash
uv run python your_script.py
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

## Paper Reproduction

The JRSS-style paper artifacts live in `refs/paperjrss/`.

Use the paper reproduction notebook to regenerate the results, figures, and PDF:

```bash
uv run jupyter notebook refs/paperjrss/paper_reproduction.ipynb
```

The notebook orchestrates the existing scripts used by the paper:

- `examples/new/benchmarks.py` for the synthetic HCM validation scenarios.
- `examples/STAR/star_baseline_and_hcm_benchmark.py --outcome math` for OLS, IV, and STAR graph-family baselines.
- `examples/STAR/star_hcm_v2_teacher_student.py` for the class-as-unit STAR HCM runs.
- `examples/STAR/star_make_paper_ready_figures.py` for all figures copied into `refs/paperjrss/Fig/`.
- `refs/paperjrss/make_performance_figures.py` for the convergence figure with runtime labels.
- `latexmk -pdf paper.tex` inside `refs/paperjrss/` for the final PDF.

Important paper outputs:

- Main TeX source: `refs/paperjrss/paper.tex`
- Compiled PDF: `refs/paperjrss/paper.pdf`
- Paper figures: `refs/paperjrss/Fig/`
- Reproduction notebook: `refs/paperjrss/paper_reproduction.ipynb`
- STAR baseline JSON: `examples/STAR/results/star_baseline_math_benchmark.json`
- STAR HCM JSON: `examples/STAR/results/star_hcm_v2_teacher_student.json`
- Parallel runtime JSON: `examples/STAR/results/ate_10_40_parallel_speed_test.json`

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.