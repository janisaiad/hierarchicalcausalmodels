#!/bin/bash
# One-liner installation and setup script for hierarchicalcausalmodels
# Explanation of each command:
# 1. Install uv package manager (official installer script, pipes output to sh for execution)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Alternative: install uv via pip (redundant if line 1 succeeds, but kept as fallback)
pip install uv
# 3. Create/clear virtual environment in .venv directory (--clear removes existing venv if present)
uv venv --clear
# 4. Activate the virtual environment (needed for subsequent commands to use venv's Python)
source .venv/bin/activate
# 5-6. Navigate to package directory (NOTE: This is incorrect - should install from root where pyproject.toml is)
# FIXED: Removed these lines - installation should happen from project root
# cd src
# cd hierarchicalcausalmodels
# 7. Install package in editable mode (-e flag) with all dependencies from pyproject.toml
# This reads pyproject.toml from current directory and installs the package
uv pip install -e .
# 8. Run test script (if it exists at tests/test_env.py relative to project root)
uv run tests/test_env.py
# 9. Re-activate venv (redundant since already activated, but harmless)
