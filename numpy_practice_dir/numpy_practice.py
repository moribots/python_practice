import numpy as np
import time
from common.test_utils import debug_print


def problem1(a, b):
    """Compute dot product of two vectors"""
    # Return dot product
    return np.dot(a, b)


def problem2(arr, rows, cols):
    """Reshape array to (rows, cols)"""
    # Return reshaped array
    return arr.reshape(rows, cols)


def problem3(A, B):
    """Compute matrix multiplication"""
    # Return matrix multiplication
    return A @ B


def problem4(matrix):
    """Find eigenvalues and eigenvectors"""
    # Return eigenvalues and eigenvectors
    return np.linalg.eig(matrix)


def problem5(matrix):
    """Compute inverse of matrix"""
    # Return matrix inverse
    return np.linalg.inv(matrix)


def problem6(shape):
    """Generate random array and compute mean"""
    # Generate random array and return mean
    arr = np.random.rand(*shape)  # * unpacks shape
    return np.mean(arr)


def problem7():
    """Broadcasting - add scalar to array and different shaped arrays"""
    # Demonstrate broadcasting with scalar and array operations
    # Scalar
    arr1 = np.array([1, 2, 3])
    res1 = arr1 + 5  # expect 6, 7, 8

    # Array
    arr1.shape = (3, 1)  # in-place reshape
    arr2 = arr1.T
    res2 = arr1 + arr2  # --> will broadcast to (3, 3)
    # a1[0]+a2[0], a1[0]+a2[1], a1[0]+a2[2]
    # a1[1]+a2[0], a1[1]+a2[1], a1[1]+a2[2]
    # a1[2]+a2[0], a1[2]+a2[1], a1[2]+a2[2]

    return res1, res2


def problem8(arr, mask_values):
    """
    Perform advanced indexing using boolean and fancy indexing.
    """
    # Create boolean mask for elements in arr that are in mask_values
    mask = np.isin(arr, mask_values)

    # Filter the array using boolean indexing
    filtered = arr[mask]

    # Get the indices of the matching elements for fancy indexing
    indices = np.where(mask)[0]

    # Apply fancy indexing to retrieve the same elements
    fancy_indexed = arr[indices]

    # Return both results as a tuple
    return filtered, fancy_indexed


def problem9(arr):
    """Vectorized operations vs loops performance comparison"""
    # Compare vectorized operations vs Python loops performance

    # Slow: Using a Python loop
    start_time = time.time()
    result_slow = []
    for x in arr:
        result_slow.append(x**2)
    slow_time = time.time() - start_time
    debug_print(f"Slow loop time: {slow_time:.6f} seconds")

    # Med: Vectorized operation
    start_time = time.time()
    result_med = arr ** 2
    med_time = time.time() - start_time
    debug_print(f"Vectorized operation time: {med_time:.6f} seconds")

    # Fast: In-place vectorized operation
    start_time = time.time()
    arr **= 2  # Use in-place to modify arr directly
    fast_time = time.time() - start_time
    debug_print(f"In-place operation time: {fast_time:.6f} seconds")


def problem10(arr):
    """Memory-efficient operations with views vs copies"""
    # Demonstrate memory-efficient operations using views
    old_arr = arr.copy()
    view = arr.view()
    # more memory efficient, will still modify arr, kind of like a reference in cpp.
    view[0] = 999
    debug_print(old_arr is not arr)

    # Check data ownership
    debug_print(arr.base is None)  # True (owns data)
    debug_print(view.base is arr)  # True (view of arr)


def problem11(A, b):
    """Solve linear system Ax = b"""
    # Use np.linalg.solve to solve linear system
    return np.linalg.solve(A, b)


def problem12(matrix):
    """
    Compute Singular Value Decomposition (SVD) of the input matrix and reconstruct it.

    SVD decomposes a matrix A into U, S, and Vt such that A ≈ U @ diag(S) @ Vt.
    - U: Left singular vectors (orthogonal matrix).
    - S: Singular values (1D array, sorted in descending order).
    - Vt: Right singular vectors (orthogonal matrix, transposed).

    For a matrix of shape (m, n), U has shape (m, k), S has shape (k,), and Vt has shape (k, n),
    where k = min(m, n). Reconstruction uses U @ np.diag(S) @ Vt to handle the 1D S array.

    Args:
        matrix (np.ndarray): Input matrix to decompose.

    Returns:
        tuple: (U, S, Vt)
    """
    # Compute SVD and reconstruct matrix
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    # NOTE: full_matrices=False returns the reduced versions of the U and Vt matrices
    # to make sure that their shapes are compatible for reconstruction.
    # This returns the minimal factors that exactly reconstruct the original matrix.
    # The extra "lost" columns are just orthonormal padding.
    debug_print("Shapes:", U.shape, np.diag(S).shape, Vt.shape)
    return U, S, Vt


def problem13(matrix):
    """
    Matrix decompositions - LU, QR, Cholesky.

    This function demonstrates three fundamental matrix decompositions:
    - LU: Decomposes PA = LU where P is permutation, L is unit lower
      triangular, U is upper triangular
    - QR: Decomposes A = QR where Q is orthogonal and R is upper triangular
    - Cholesky: Decomposes A = LL^T for symmetric positive definite matrices

    Args:
        matrix (np.ndarray): Input square matrix to decompose

    Returns:
        dict: Dictionary containing decomposition results with keys:
            - 'lu': tuple (P, L, U) from LU decomposition with partial pivoting
            - 'qr': tuple (Q, R) from QR decomposition
            - 'cholesky': L matrix from Cholesky decomposition (if applicable)
            - 'residuals': dictionary of reconstruction residuals for
              validation
    """
    decompositions = {}
    residuals = {}

    # LU Decomposition with partial pivoting
    # scipy.linalg.lu returns P, L, U such that PA = LU
    from scipy.linalg import lu, qr, cholesky
    P, L, U = lu(matrix)
    decompositions['lu'] = (P, L, U)

    # Compute LU residual: ||PA - LU||_F / ||A||_F
    lu_reconstruction = P @ matrix
    lu_product = L @ U
    residuals['lu'] = (np.linalg.norm(lu_reconstruction - lu_product, 'fro') /
                       np.linalg.norm(matrix, 'fro'))

    # QR Decomposition
    # Uses Householder reflections for numerical stability
    Q, R = qr(matrix)
    decompositions['qr'] = (Q, R)

    # Compute QR residual: ||A - QR||_F / ||A||_F
    qr_reconstruction = Q @ R
    residuals['qr'] = (np.linalg.norm(matrix - qr_reconstruction, 'fro') /
                       np.linalg.norm(matrix, 'fro'))

    # Cholesky Decomposition (only for symmetric positive definite matrices)
    try:
        # Check if matrix is symmetric positive definite
        is_symmetric = np.allclose(matrix, matrix.T)
        is_positive_def = np.all(np.linalg.eigvals(matrix) > 0)
        if is_symmetric and is_positive_def:
            L_chol = cholesky(matrix, lower=True)
            decompositions['cholesky'] = L_chol

            # Compute Cholesky residual: ||A - LL^T||_F / ||A||_F
            chol_reconstruction = L_chol @ L_chol.T
            residuals['cholesky'] = (
                np.linalg.norm(matrix - chol_reconstruction, 'fro') /
                np.linalg.norm(matrix, 'fro'))
        else:
            decompositions['cholesky'] = None
            residuals['cholesky'] = None
            debug_print("Matrix is not symmetric positive definite - "
                        "Cholesky not applicable")
    except np.linalg.LinAlgError:
        decompositions['cholesky'] = None
        residuals['cholesky'] = None
        debug_print("Cholesky decomposition failed - "
                    "matrix not positive definite")

    # Add residuals for validation
    decompositions['residuals'] = residuals

    # Debug output for educational purposes
    debug_print(f"LU residual: {residuals['lu']:.2e}")
    debug_print(f"QR residual: {residuals['qr']:.2e}")
    if residuals['cholesky'] is not None:
        debug_print(f"Cholesky residual: {residuals['cholesky']:.2e}")

    return decompositions


def problem14(f, a, b, n=1000):
    """
    Numerical integration using the trapezoidal rule.

    The trapezoidal rule approximates the integral ∫[a,b] f(x) dx by dividing
    the interval into n subintervals and approximating the area under the curve
    with trapezoids. The formula is:

    ∫[a,b] f(x) dx ≈ (h/2) * [f(a) + 2*sum(f(xi)) + f(b)]

    where h = (b-a)/n and xi are the interior points.

    Args:
        f (callable): Function to integrate
        a (float): Lower bound of integration
        b (float): Upper bound of integration
        n (int): Number of subintervals (default 1000)

    Returns:
        float: Approximate value of the integral
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)  # n+1
    y = f(x)
    # f(a) = y[0], f(b) = y[-1]
    # sum of interior points = sum(y[1:-1])
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral


def problem15(signal):
    """
    FFT (Fast Fourier Transform) for signal processing.

    This demonstrates the fundamental FFT operations for signal analysis:
    - Forward FFT: converts time-domain signal to frequency-domain
    - Inverse FFT: converts frequency-domain back to time-domain

    The FFT decomposes a signal into its constituent frequencies, enabling
    frequency analysis, filtering, and other signal processing operations.

    Args:
        signal (np.ndarray): Input time-domain signal

    Returns:
        tuple: (fft_result, ifft_result) where:
            - fft_result: Complex frequency-domain representation
            - ifft_result: Reconstructed time-domain signal (should match
              input)
    """
    # Forward FFT
    fft_result = np.fft.fft(signal)

    # Inverse FFT
    ifft_result = np.fft.ifft(fft_result)

    return fft_result, ifft_result
