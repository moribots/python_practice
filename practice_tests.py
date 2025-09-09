import numpy as np
import torch
import pandas as pd
import einops
from einops_practice import *
from python_practice import *
from pandas_practice import *
from pytorch_practice import *
from numpy_practice import *
from reinforcement_learning_practice import *
from robotics_fundamentals_practice import *

# Test functions


def test_einops():
    # Problem 1
    tensor = np.random.rand(2, 10, 10, 3)
    expected = einops.rearrange(tensor, 'b h w c -> b c h w')
    try:
        result = problem1(tensor)
        assert np.allclose(result, expected)
        print("✅ Problem 1 (Einops) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (Einops) Failed: {e}")

    # Problem 2
    tensor = np.random.rand(2, 5, 10)
    expected = einops.rearrange(tensor, 'b s h -> b (s h)')
    try:
        result = problem2(tensor)
        assert np.allclose(result, expected)
        print("✅ Problem 2 (Einops) Passed")
    except Exception as e:
        print(f"❌ Problem 2 (Einops) Failed: {e}")

    # Add more tests similarly...


def test_python():
    # Problem 1
    arr = [1, 2, 3, 4, 5]
    target = 3
    try:
        result = problem1(arr, target)
        assert result == 2
        print("✅ Problem 1 (Python) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (Python) Failed: {e}")

    # Add more...


def test_pandas():
    # Problem 1
    df = pd.DataFrame({'a': [1, 2, 3]})
    try:
        result = problem1('dummy.csv', 'a')  # Mock
        print("✅ Problem 1 (Pandas) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (Pandas) Failed: {e}")

    # Add more...


def test_pytorch():
    # Problem 1
    model = Problem1(10, 5, 1)
    x = torch.randn(1, 10)
    try:
        out = model(x)
        assert out.shape == (1, 1)
        print("✅ Problem 1 (PyTorch) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (PyTorch) Failed: {e}")

    # Add more...


def test_numpy():
    # Problem 1
    a = np.array([1, 2])
    b = np.array([3, 4])
    try:
        result = problem1(a, b)
        assert result == 11
        print("✅ Problem 1 (NumPy) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (NumPy) Failed: {e}")

    # Add more...


def test_rl():
    # Problem 1
    Q = np.zeros((2, 2))
    try:
        problem1(Q, 0, 0, 1, 1, 0.1, 0.9)
        print("✅ Problem 1 (RL) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (RL) Failed: {e}")

    # Add more...


def test_robotics():
    # Problem 1
    try:
        result = problem1(0, 0, 1, 1)
        assert np.allclose(result, (2, 0))
        print("✅ Problem 1 (Robotics) Passed")
    except Exception as e:
        print(f"❌ Problem 1 (Robotics) Failed: {e}")

    # Add more...


if __name__ == "__main__":
    test_einops()
    test_python()
    test_pandas()
    test_pytorch()
    test_numpy()
    test_rl()
    test_robotics()
