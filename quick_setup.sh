#!/bin/bash

# Quick Setup Script for Python Practice
# This script uses conda run (doesn't require conda init)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

ENV_NAME="python_practice"

echo
print_info "🚀 Quick Python Practice Setup"
echo "==============================="

# Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME}"; then
    print_info "Creating conda environment..."
    conda create -n ${ENV_NAME} python=3.10 -y
    conda run -n ${ENV_NAME} pip install -r requirements.txt
    print_status "Environment created and packages installed"
else
    print_status "Environment already exists"
fi

# Test the environment
print_info "Testing environment..."
conda run -n ${ENV_NAME} python -c "import numpy, pandas, torch, einops; print('✅ All core packages working')"

echo
print_status "Setup complete!"
print_info "To activate the environment manually: conda activate ${ENV_NAME}"
print_info "To run tests: conda run -n ${ENV_NAME} python practice_tests.py"
echo
