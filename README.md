# Python Practice for Robotics

This repository contains Python practice files covering key topics for Robotics Research Engineer roles, including einops, Python algorithms, pandas, PyTorch, NumPy, reinforcement learning, and robotics fundamentals.

## Project Structure
```
├── einops/
│   ├── __init__.py
│   ├── einops_practice.py
│   └── README.md
├── python/
│   ├── __init__.py
│   ├── python_practice.py
│   └── README.md
├── pandas/
│   ├── __init__.py
│   ├── pandas_practice.py
│   └── README.md
├── pytorch/
│   ├── __init__.py
│   ├── pytorch_practice.py
│   └── README.md
├── numpy/
│   ├── __init__.py
│   ├── numpy_practice.py
│   └── README.md
├── reinforcement_learning/
│   ├── __init__.py
│   ├── reinforcement_learning_practice.py
│   └── README.md
├── robotics_fundamentals/
│   ├── __init__.py
│   ├── robotics_fundamentals_practice.py
│   └── README.md
├── practice_tests.py
├── requirements.txt
└── README.md
```

## Topics Covered
- **einops**: Tensor rearrangement and manipulation
- **Python**: Algorithms, data structures, recursion
- **pandas**: Data manipulation and analysis
- **PyTorch**: Neural networks, tensors, autograd
- **NumPy**: Arrays, linear algebra, matrix operations
- **Reinforcement Learning**: Q-learning, policy evaluation, Monte Carlo
- **Robotics Fundamentals**: Kinematics, control, planning, perception

## How to Run

### 1. Set Up Environment
Create a Conda environment and install dependencies:
```bash
conda create -n python_practice python=3.10
conda activate python_practice
pip install -r requirements.txt
```

### 2. Study Individual Topics
Each topic has its own folder with:
- Practice problems file
- Study guide README with worked examples

Navigate to any topic folder and read the README.md for concepts and examples:
```bash
cd einops
cat README.md
```

### 3. Run Practice Files
Execute any practice file directly:
```bash
python einops/einops_practice.py
```
Note: Files contain placeholder functions that raise `NotImplementedError`. Implement the TODOs before running.

### 4. Run Tests
Run the test suite to check implementations:
```bash
python practice_tests.py
```
Output shows:
- ✅ Problem X (Topic) Passed: Implementation correct
- ❌ Problem X (Topic) Failed: Implementation has errors
- ⚠️ Problem X (Topic) Not Implemented: Function not yet filled

Problems are numbered dynamically based on implemented functions (skips unimplemented ones).

### 5. Implement Problems
1. Open a practice file (e.g., `einops/einops_practice.py`)
2. Replace `raise NotImplementedError` with your solution
3. Run `practice_tests.py` to verify
4. Repeat for other problems

## Adding New Questions
To add new problems:
1. Choose the appropriate topic folder
2. Add a new function `def problemN(...):` with TODO and `raise NotImplementedError` in the practice file
3. Add corresponding test case in `practice_tests.py` under the relevant `test_*` function
4. Update the folder's README.md with explanation and example
5. Follow the pattern: try-except with counter for dynamic numbering

## Tips
- Start with the study guides in each folder's README.md
- Focus on problems relevant to robotics: perception, planning, control, ML for embodied AI
- Practice with real data where possible
- Use the test suite for immediate feedback
- Review Robotics papers on robotics and AI for context

## Dependencies
- einops
- pandas
- torch
- numpy
- gym (for RL)

Install via `pip install -r requirements.txt`.
