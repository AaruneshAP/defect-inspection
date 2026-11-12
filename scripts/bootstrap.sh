#!/usr/bin/env bash
<<'DOCSTRING'
Purpose: Initializes the defect-inspection project environment, creating the Python virtual environment and installing required dependencies.
Inputs: None (reads requirements.txt in the workspace root).
Outputs: Installs virtual environment in .venv with all project dependencies.
DOCSTRING

set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data/raw data/processed models notebooks
echo "Bootstrap completed successfully."

