from . import numpy_practice as npp
import numpy as np
from common.test_utils import _pass, _fail


def test_numpy():
    print("\nTesting NumPy Problems:")
    counter = 1
    # Problem 1
    a = np.array([1, 2])
    b = np.array([3, 4])
    try:
        result = npp.problem1(a, b)
        assert result == 11
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 2
    arr = np.arange(12)
    try:
        result = npp.problem2(arr, 3, 4)
        assert result.shape == (3, 4)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 3
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    try:
        result = npp.problem3(A, B)
        expected = A @ B
        assert np.allclose(result, expected)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 4
    matrix = np.array([[4, 0], [0, 3]])
    try:
        eigvals, eigvecs = npp.problem4(matrix)
        assert np.allclose(eigvals, [4, 3])
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 5
    matrix = np.array([[1, 2], [3, 4]])
    try:
        result = npp.problem5(matrix)
        expected = np.linalg.inv(matrix)
        assert np.allclose(result, expected)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 6
    shape = (10, 10)
    try:
        result = npp.problem6(shape)
        assert isinstance(result, float)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 7
    try:
        res1, res2 = npp.problem7()
        assert res1.shape == (3,) and res2.shape == (3, 3)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 8
    arr = np.array([1, 5, 2, 8, 3, 9, 4])
    mask_values = [2, 8]
    try:
        result = npp.problem8(arr, mask_values)
        # Should return filtered and fancy indexed arrays
        assert isinstance(result, tuple)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 9
    arr = np.random.rand(1000)
    try:
        npp.problem9(arr)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 10
    arr = np.random.rand(10, 10)
    try:
        npp.problem10(arr)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 11
    A = np.array([[2, 1], [1, 3]])
    b = np.array([3, 4])
    try:
        result = npp.problem11(A, b)
        expected = np.linalg.solve(A, b)
        assert np.allclose(result, expected)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 12
    matrix = np.random.rand(10, 8)
    try:
        U, s, Vt = npp.problem12(matrix)
        reconstructed = U @ np.diag(s) @ Vt
        assert np.allclose(matrix, reconstructed, atol=1e-10)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 13
    matrix = np.random.rand(5, 5)
    try:
        decompositions = npp.problem13(matrix)
        assert isinstance(decompositions, dict)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 14
    def test_func(x):
        return x**2
    try:
        integral = npp.problem14(test_func, 0, 1)
        expected = 1/3  # Integral of x^2 from 0 to 1
        assert abs(integral - expected) < 0.01
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1

    # Problem 15
    signal = np.random.rand(100)
    try:
        fft_result, ifft_result = npp.problem15(signal)
        assert np.allclose(signal, ifft_result.real, atol=1e-10)
        _pass(counter, "NumPy")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "NumPy", e)
        counter += 1
