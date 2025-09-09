#!/bin/bash

# Repository Setup Script for Python Practice
# This script activates the conda environment and sets up the development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_info "Please install Miniconda or Anaconda first:"
        print_info "  https://docs.conda.io/projects/miniconda/en/latest/"
        exit 1
    fi
    print_status "Conda found: $(conda --version)"
}

# Check if the environment exists
check_environment() {
    local env_name="python_practice"

    if ! conda env list | grep -q "^${env_name}"; then
        print_warning "Conda environment '${env_name}' not found"
        print_info "Creating environment from environment.yml or requirements.txt..."

        # Try to create from environment.yml first
        if [ -f "environment.yml" ]; then
            print_info "Found environment.yml, creating environment..."
            conda env create -f environment.yml
        elif [ -f "requirements.txt" ]; then
            print_info "Found requirements.txt, creating environment..."
            conda create -n ${env_name} python=3.10 -y
            conda run -n ${env_name} pip install -r requirements.txt
        else
            print_error "No environment.yml or requirements.txt found"
            print_info "Please create one of these files or manually create the environment:"
            print_info "  conda create -n ${env_name} python=3.10"
            exit 1
        fi
    fi

    print_status "Conda environment '${env_name}' is ready"
}

# Activate the environment
activate_environment() {
    local env_name="python_practice"

    print_info "Activating conda environment: ${env_name}"

    # Try to activate using conda activate (preferred method)
    if conda activate ${env_name} 2>/dev/null; then
        print_status "Successfully activated ${env_name} using conda activate"
    else
        print_warning "conda activate failed. This might be because conda init hasn't been run."
        print_info "You can either:"
        print_info "  1. Run: conda init bash (then restart terminal)"
        print_info "  2. Or manually activate: conda activate ${env_name}"
        print_info ""
        print_info "For now, the environment has been created and packages installed."
        print_info "You can activate it manually when ready."
        return 1
    fi

    # Verify activation
    if [ "$CONDA_DEFAULT_ENV" = "${env_name}" ]; then
        print_status "Environment verification successful"
        print_info "Python: $(python --version)"
        print_info "Current environment: $CONDA_DEFAULT_ENV"
    else
        print_warning "Environment activation may not be complete"
        print_info "Current CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    fi
}

# Setup development environment
setup_dev_environment() {
    print_info "Setting up development environment..."

    # Check if requirements are installed
    if [ -f "requirements.txt" ]; then
        print_info "Checking if all requirements are installed..."
        # This is a simple check - you might want to make it more sophisticated
        python -c "import numpy, pandas, torch, einops, gym" 2>/dev/null && \
            print_status "Core dependencies are available" || \
            print_warning "Some dependencies might be missing. Run: pip install -r requirements.txt"
    fi

    # Check if we're in the right directory
    if [ -f "practice_tests.py" ]; then
        print_status "Repository structure looks correct"
    else
        print_warning "practice_tests.py not found. Are you in the right directory?"
    fi
}

# Main execution
main() {
    echo
    print_info "🚀 Python Practice Repository Setup"
    echo "=================================="

    check_conda
    check_environment
    activate_environment
    setup_dev_environment

    echo
    print_status "Setup complete! You're ready to start practicing."
    print_info "Try running: python practice_tests.py"
    echo
}

# Run main function
main "$@"
