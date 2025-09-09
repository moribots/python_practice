# Python Practice Environment Setup

This repository includes automated setup scripts to get your development environment ready quickly.

## Quick Setup (Recommended)

Run the quick setup script:
```bash
./quick_setup.sh
```

This will:
- ✅ Create the `python_practice` conda environment (if it doesn't exist)
- ✅ Install all required packages from `requirements.txt`
- ✅ Verify that core packages are working
- ✅ Provide instructions for manual activation

## Full Setup (Advanced)

For a more comprehensive setup that attempts automatic activation:
```bash
./repo_setup.sh
```

**Note:** This requires `conda init` to have been run previously. If you get activation errors, use the quick setup instead.

## Manual Environment Management

### Activate Environment
```bash
conda activate python_practice
```

### Run Tests
```bash
python practice_tests.py
```

Or without activating:
```bash
conda run -n python_practice python practice_tests.py
```

### Deactivate Environment
```bash
conda deactivate
```

## Troubleshooting

### Conda Init Issues
If you get "conda init" errors:
1. Run: `conda init bash` (then restart your terminal)
2. Or use the quick setup script which doesn't require conda init

### Environment Already Exists
If you need to recreate the environment:
```bash
conda env remove -n python_practice
./quick_setup.sh
```

### Package Installation Issues
To reinstall all packages:
```bash
conda run -n python_practice pip install -r requirements.txt
```

## Development Workflow

1. **First time setup:** `./quick_setup.sh`
2. **Daily development:** `conda activate python_practice`
3. **Run tests:** `python practice_tests.py`
4. **Work on problems:** Edit the TODO sections in practice files
5. **Check progress:** Run tests to see which problems are solved

Happy practicing! 🚀
