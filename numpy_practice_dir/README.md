# NumPy Study Guide

NumPy is fundamental for numerical computing in Python, essential for robotics calculations, matrix operations, and data preprocessing. Robotics interviews often test numerical computing skills and linear algebra knowledge.

## Key Concepts

### Arrays
- `np.array()`: Create arrays
- `arr.shape`: Get dimensions
- `arr.reshape()`: Change shape
- `arr.T`: Transpose

### Operations
- Element-wise: `arr1 + arr2`
- Matrix multiplication: `np.dot()` or `arr1 @ arr2`
- Broadcasting: Automatic shape matching

### Linear Algebra
- `np.linalg.inv()`: Matrix inverse
- `np.linalg.eig()`: Eigenvalues/eigenvectors
- `np.linalg.det()`: Determinant

### Random
- `np.random.rand()`: Uniform [0,1)
- `np.random.randn()`: Standard normal
- `np.random.randint()`: Random integers

## Interview-Ready Concepts

### Broadcasting Rules
- Arrays can be broadcasted if dimensions are compatible
- Rightmost dimensions must match or be 1
- Critical for memory-efficient operations

### Memory Layout
- C-order (row-major) vs F-order (column-major)
- `arr.flags.c_contiguous`: Check memory layout
- Affects performance in different operations

### Vectorization
- Replace loops with array operations
- `np.vectorize()` for element-wise functions
- Performance gains with large arrays

## Worked Examples

### Problem 1: Dot Product
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)
print(result)  # 32
```

### Problem 2: Reshape Array
```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]
reshaped = arr.reshape(3, 4)
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

### Problem 3: Matrix Multiplication
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A @ B
print(result)
# [[19 22]
#  [43 50]]
```

### Problem 4: Eigenvalues and Eigenvectors
```python
matrix = np.array([[4, 0], [0, 3]])
eigenvals, eigenvecs = np.linalg.eig(matrix)
print("Eigenvalues:", eigenvals)  # [4. 3.]
print("Eigenvectors:", eigenvecs)
```

### Problem 5: Matrix Inverse
```python
matrix = np.array([[1, 2], [3, 4]])
inverse = np.linalg.inv(matrix)
print(inverse)
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

### Problem 6: Random Array Mean
```python
random_array = np.random.rand(10, 10)
mean_value = np.mean(random_array)
print(mean_value)  # Around 0.5
```

## Advanced Interview Topics

### Broadcasting Examples
```python
# Add scalar to array
arr = np.array([1, 2, 3])
result = arr + 10  # Broadcasting scalar
print(result)  # [11 12 13]

# Add different shaped arrays
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[4], [5], [6]])  # (3, 1)
result = A + B  # Broadcasting to (3, 3)
print(result)
# [[5 6 7]
#  [6 7 8]
#  [7 8 9]]
```

### Advanced Indexing
```python
# Boolean indexing
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print(arr[mask])  # [4 5]

# Fancy indexing
indices = [0, 2, 4]
print(arr[indices])  # [1 3 5]

# Multi-dimensional indexing
matrix = np.arange(12).reshape(3, 4)
print(matrix[1, :])  # Second row
print(matrix[:, 2])  # Third column
```

### Performance Optimization
```python
# Vectorized operations are faster than loops
arr = np.random.rand(1000000)

# Slow way
result_slow = []
for x in arr:
    result_slow.append(x ** 2)

# Fast way
result_fast = arr ** 2

# Even faster with in-place operations
arr **= 2
```

### Memory-Efficient Operations
```python
# Use views instead of copies when possible
arr = np.arange(12)
view = arr[::2]  # View, not copy
view[0] = 999
print(arr)  # [999, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Check if array owns its data
print(arr.base is None)  # True (owns data)
print(view.base is arr)  # True (view of arr)
```

### Linear Algebra Applications
```python
# Solve linear system Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 7])
x = np.linalg.solve(A, b)
print(x)  # [1. 2.]

# Singular value decomposition
matrix = np.random.rand(5, 3)
U, s, Vt = np.linalg.svd(matrix)
print("Singular values:", s)
```

## Practice Tips
- Use vectorized operations instead of loops
- Understand broadcasting rules
- Check array shapes with `.shape`
- Use `np.newaxis` for dimension expansion
- NumPy operations are much faster than Python loops
- Consider memory layout for performance-critical code
- Use `np.allclose()` for floating-point comparisons
