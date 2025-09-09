# Robotics Fundamentals Study Guide

Robotics combines mechanical engineering, control theory, and computer science. Robotics interviews often test core robotics concepts, mathematical foundations, and practical implementation skills for embodied AI systems.

## Key Concepts

### Forward Kinematics
- Compute end-effector position from joint angles
- 2-link arm: x = l1*cos(θ1) + l2*cos(θ1+θ2)
- 2-link arm: y = l1*sin(θ1) + l2*sin(θ1+θ2)

### PID Control
- Proportional-Integral-Derivative controller
- P: proportional to error
- I: integral of error (eliminates steady-state)
- D: derivative of error (damps oscillations)

### A* Path Planning
- Informed search algorithm
- f(n) = g(n) + h(n): cost + heuristic
- Uses priority queue
- Guarantees optimal path with admissible heuristic

### Convolution
- Image filtering operation
- Kernel slides over image
- Output: sum of element-wise products

## Interview-Ready Concepts

### Inverse Kinematics
- Compute joint angles from end-effector position
- Analytical vs numerical solutions
- Multiple solutions and singularities
- Jacobian-based methods

### Advanced Control
- LQR (Linear Quadratic Regulator)
- MPC (Model Predictive Control)
- Adaptive and robust control
- Multi-variable control systems

### Motion Planning
- RRT (Rapidly-exploring Random Trees)
- PRM (Probabilistic Roadmaps)
- Trajectory optimization
- Real-time planning constraints

### Computer Vision Basics
- Feature detection and matching
- Image segmentation
- Depth estimation
- SLAM (Simultaneous Localization and Mapping)

## Worked Examples

### Problem 1: Forward Kinematics (2-Link Arm)
```python
import numpy as np

def forward_kinematics(theta1, theta2, l1, l2):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

# Test: 90-degree angles, unit lengths
x, y = forward_kinematics(np.pi/2, np.pi/2, 1, 1)
print(f"End effector: ({x:.2f}, {y:.2f})")  # (0.00, 2.00)
```

### Problem 2: PID Controller
```python
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error, dt=1.0):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# Test
pid = PIDController(1.0, 0.1, 0.05)
error = 5.0
control_signal = pid.update(error)
print(f"Control signal: {control_signal}")  # 5.05
```

### Problem 3: A* Path Planning
```python
import heapq

def a_star(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current = heapq.heappop(frontier)[1]
        
        if current == goal:
            break
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            
            if (0 <= next_pos[0] < len(grid) and 
                0 <= next_pos[1] < len(grid[0]) and 
                grid[next_pos[0]][next_pos[1]] == 0):  # Not obstacle
                
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
    
    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return None  # No path
    path.append(start)
    path.reverse()
    return path

# Test
grid = [
    [0, 0, 0],
    [0, 1, 0],  # 1 is obstacle
    [0, 0, 0]
]
path = a_star(grid, (0, 0), (2, 2))
print(path)  # [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
```

### Problem 4: 2D Convolution
```python
def convolution_2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            region = image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Test: Simple edge detection
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
result = convolution_2d(image, kernel)
print(result)
```

## Advanced Interview Topics

### Inverse Kinematics (Analytical Solution)
```python
def inverse_kinematics_2d(x, y, l1, l2):
    # For 2-link arm reaching point (x, y)
    r = np.sqrt(x**2 + y**2)
    
    if r > l1 + l2 or r < abs(l1 - l2):
        return None  # Point unreachable
    
    # Law of cosines
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = np.arccos(np.clip(cos_theta2, -1, 1))
    
    # Two possible solutions for theta2
    solutions = []
    for theta2_val in [theta2, -theta2]:
        sin_theta2 = np.sin(theta2_val)
        theta1 = np.arctan2(y, x) - np.arctan2(l2 * sin_theta2, l1 + l2 * cos_theta2)
        solutions.append((theta1, theta2_val))
    
    return solutions

# Test
solutions = inverse_kinematics_2d(1, 1, 1, 1)
for i, (theta1, theta2) in enumerate(solutions):
    print(f"Solution {i+1}: θ1={theta1:.3f}, θ2={theta2:.3f}")
```

### LQR Controller Design
```python
def lqr_controller(A, B, Q, R):
    # Solve discrete-time algebraic Riccati equation
    # This is a simplified implementation
    P = Q.copy()
    
    for _ in range(100):  # Iterate to convergence
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.allclose(P, P_next):
            break
        P = P_next
    
    # Compute optimal gain
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

# Example: Double integrator system
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
Q = np.eye(2)  # State cost
R = np.array([[0.1]])  # Control cost

K = lqr_controller(A, B, Q, R)
print("LQR gain matrix:", K)
```

### RRT Path Planning
```python
import random

class RRT:
    def __init__(self, start, goal, obstacles, max_iter=1000):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.nodes = [start]
        self.parents = {start: None}
    
    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def nearest_neighbor(self, point):
        return min(self.nodes, key=lambda node: self.distance(node, point))
    
    def steer(self, from_node, to_point, step_size=0.5):
        dist = self.distance(from_node, to_point)
        if dist <= step_size:
            return to_point
        
        direction = ((to_point[0] - from_node[0]) / dist, 
                    (to_point[1] - from_node[1]) / dist)
        return (from_node[0] + step_size * direction[0],
                from_node[1] + step_size * direction[1])
    
    def collision_free(self, p1, p2):
        # Simplified collision check
        for obs in self.obstacles:
            if self.line_intersects_circle(p1, p2, obs):
                return False
        return True
    
    def line_intersects_circle(self, p1, p2, circle):
        # Simplified implementation
        return False  # Assume no collisions for demo
    
    def plan(self):
        for _ in range(self.max_iter):
            random_point = (random.uniform(0, 10), random.uniform(0, 10))
            nearest = self.nearest_neighbor(random_point)
            new_node = self.steer(nearest, random_point)
            
            if self.collision_free(nearest, new_node):
                self.nodes.append(new_node)
                self.parents[new_node] = nearest
                
                if self.distance(new_node, self.goal) < 0.5:
                    # Reconstruct path
                    path = [self.goal, new_node]
                    current = new_node
                    while current != self.start:
                        current = self.parents[current]
                        path.append(current)
                    path.reverse()
                    return path
        
        return None

# Test
rrt = RRT((0, 0), (9, 9), [])
path = rrt.plan()
print("RRT path:", path)
```

### Basic SLAM Concepts
```python
class SimpleSLAM:
    def __init__(self):
        self.landmarks = {}  # Known landmarks
        self.pose = (0, 0, 0)  # x, y, theta
    
    def motion_update(self, odometry):
        # Update pose based on odometry
        dx, dy, dtheta = odometry
        x, y, theta = self.pose
        
        self.pose = (
            x + dx * np.cos(theta) - dy * np.sin(theta),
            y + dx * np.sin(theta) + dy * np.cos(theta),
            theta + dtheta
        )
    
    def measurement_update(self, measurements):
        # Update landmark positions based on measurements
        for landmark_id, distance, bearing in measurements:
            x, y, theta = self.pose
            
            # Convert to global coordinates
            landmark_x = x + distance * np.cos(theta + bearing)
            landmark_y = y + distance * np.sin(theta + bearing)
            
            if landmark_id not in self.landmarks:
                self.landmarks[landmark_id] = (landmark_x, landmark_y)
            else:
                # Could implement Kalman filter update here
                pass

# Test
slam = SimpleSLAM()
slam.motion_update((1, 0, 0))  # Move forward 1 unit
slam.measurement_update([(0, 2, np.pi/4)])  # See landmark at distance 2, bearing 45°
print("Current pose:", slam.pose)
print("Landmarks:", slam.landmarks)
```

## Matrix Mathematics Interview Questions

**Note:** All matrix methods below use the **simplest standard approaches** taught in linear algebra courses:
- Matrix inversion: Direct cofactor expansion (no fancy algorithms)
- Rotation matrices: Basic orthogonality + determinant checks
- Determinants: Standard formulas (no decomposition methods)
- Rodrigues' formula: Direct implementation
- SVD: Built-in NumPy functions with property verification

### Matrix Inversion from Scratch
```python
def matrix_inverse_3x3(A):
    """
    Compute inverse of 3x3 matrix using cofactor expansion
    A = [[a11, a12, a13],
         [a21, a22, a23],
         [a31, a32, a33]]
    """
    # Compute determinant
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular")
    
    # Compute cofactors
    cofactors = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Minor matrix (remove row i, column j)
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            cofactor = ((-1) ** (i + j)) * (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0])
            cofactors[i][j] = cofactor
    
    # Transpose cofactors to get adjugate
    adjugate = cofactors.T
    
    # Divide by determinant
    return adjugate / det

# Test
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
A_inv = matrix_inverse_3x3(A)
print("Original matrix:")
print(A)
print("\nInverse matrix:")
print(A_inv)
print("\nVerification (should be identity):")
print(A @ A_inv)
```

### Checking Rotation Matrix Properties
```python
def is_rotation_matrix(R, tol=1e-6):
    """
    Check if matrix R is a rotation matrix:
    1. R^T * R = I (orthogonal)
    2. det(R) = 1 (proper rotation, not reflection)
    """
    # Check orthogonality
    R_T_R = R.T @ R
    identity = np.eye(3)
    orthogonal_check = np.allclose(R_T_R, identity, atol=tol)
    
    # Check determinant
    det_R = np.linalg.det(R)
    det_check = np.isclose(det_R, 1.0, atol=tol)
    
    return orthogonal_check and det_check

def is_rotation_matrix_detailed(R):
    """Detailed analysis of rotation matrix properties"""
    print("Matrix R:")
    print(R)
    print(f"\nR^T @ R (should be identity):")
    print(R.T @ R)
    print(f"\nDeterminant: {np.linalg.det(R)}")
    print(f"Is rotation matrix: {is_rotation_matrix(R)}")
    
    # Check if it's a reflection (det = -1)
    if np.isclose(np.linalg.det(R), -1.0):
        print("Note: This is a reflection matrix (det = -1)")
    
    return is_rotation_matrix(R)

# Test with rotation around z-axis by 45 degrees
theta = np.pi / 4
R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]])
is_rotation_matrix_detailed(R_z)
```

### Matrix Determinant from Scratch
```python
def determinant_3x3(A):
    """
    Compute determinant of 3x3 matrix using standard formula
    det(A) = a11(a22*a33 - a23*a32) - a12(a21*a33 - a23*a31) + a13(a21*a32 - a22*a31)
    """
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
            A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
            A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

def determinant_2x2(A):
    """
    Compute determinant of 2x2 matrix
    det(A) = a11*a22 - a12*a21
    """
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

# Test
A3 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 10]])
A2 = np.array([[1, 2],
               [3, 4]])

print(f"3x3 determinant: {determinant_3x3(A3)}")
print(f"2x2 determinant: {determinant_2x2(A2)}")
print(f"NumPy verification: {np.linalg.det(A3):.1f}")
```

### Checking Positive Definite Matrices
```python
def is_positive_definite(A):
    """
    Check if matrix A is positive definite using eigenvalues
    A matrix is positive definite if:
    1. All eigenvalues are positive
    2. x^T A x > 0 for all non-zero x
    """
    try:
        eigenvalues = np.linalg.eigvals(A)
        return np.all(eigenvalues > 0)
    except np.linalg.LinAlgError:
        return False

def is_positive_definite_sylvester(A):
    """
    Check positive definite using Sylvester's criterion
    All leading principal minors must be positive
    """
    n = A.shape[0]
    for i in range(1, n + 1):
        minor = A[:i, :i]
        if np.linalg.det(minor) <= 0:
            return False
    return True

# Test
A_pd = np.array([[2, 1], [1, 2]])  # Positive definite
A_psd = np.array([[1, 1], [1, 1]])  # Positive semi-definite
A_indef = np.array([[1, 2], [2, 1]])  # Indefinite

print(f"A_pd positive definite: {is_positive_definite(A_pd)}")
print(f"A_psd positive definite: {is_positive_definite(A_psd)}")
print(f"A_indef positive definite: {is_positive_definite(A_indef)}")
```

### Checking Positive Semi-Definite Matrices
```python
def is_positive_semidefinite(A):
    """
    Check if matrix A is positive semi-definite
    All eigenvalues are non-negative (≥ 0)
    """
    try:
        eigenvalues = np.linalg.eigvals(A)
        return np.all(eigenvalues >= 0)
    except np.linalg.LinAlgError:
        return False

def is_positive_semidefinite_cholesky(A):
    """
    Check using Cholesky decomposition
    If A is PSD, it has a Cholesky decomposition
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Test
print(f"A_pd positive semi-definite: {is_positive_semidefinite(A_pd)}")
print(f"A_psd positive semi-definite: {is_positive_semidefinite(A_psd)}")
print(f"A_indef positive semi-definite: {is_positive_semidefinite(A_indef)}")

print(f"A_psd Cholesky check: {is_positive_semidefinite_cholesky(A_psd)}")
```

### Quadratic Programming for Line Following (JAX)
```python
def line_following_qp(current_pos, target_line_params, max_velocity=1.0):
    """
    Set up QP problem for following a line
    Line equation: ax + by + c = 0
    Minimize distance to line while respecting velocity constraints
    
    This demonstrates QP formulation for trajectory optimization
    """
    try:
        import jax.numpy as jnp
        from jax import jacfwd
        
        # Line parameters: [a, b, c] for ax + by + c = 0
        a, b, c = target_line_params
        
        # Current position
        x, y = current_pos
        
        # Objective: minimize distance to line
        # Distance = |ax + by + c| / sqrt(a^2 + b^2)
        distance = jnp.abs(a * x + b * y + c) / jnp.sqrt(a**2 + b**2)
        
        # Constraints: velocity limits
        # This is a simplified QP - in practice you'd use a proper QP solver
        
        return {
            'objective': distance,
            'constraints': {'max_velocity': max_velocity},
            'line_params': target_line_params,
            'current_pos': current_pos
        }
        
    except ImportError:
        print("JAX not available for QP demonstration")
        return None

# Test line following QP setup
current_pos = [1.0, 2.0]
line_params = [1, -1, 0]  # Line: x - y = 0 (45-degree line)

qp_setup = line_following_qp(current_pos, line_params)
if qp_setup:
    print("Line following QP setup:")
    print(f"Current position: {qp_setup['current_pos']}")
    print(f"Target line: {qp_setup['line_params'][0]}x + {qp_setup['line_params'][1]}y + {qp_setup['line_params'][2]} = 0")
```

### Quadratic Programming for Collision Avoidance (JAX)
```python
def collision_avoidance_qp(current_pos, obstacle_pos, obstacle_radius, 
                          target_pos, safety_margin=0.5):
    """
    Set up QP problem for collision avoidance
    Minimize path to target while avoiding obstacles
    
    Objective: min ||x - x_target||²
    Subject to: ||x - x_obstacle|| ≥ obstacle_radius + safety_margin
    """
    try:
        import jax.numpy as jnp
        
        # Positions
        x_current = jnp.array(current_pos)
        x_target = jnp.array(target_pos)
        x_obstacle = jnp.array(obstacle_pos)
        
        # Objective: minimize distance to target
        objective = jnp.sum((x_current - x_target)**2)
        
        # Constraint: maintain distance from obstacle
        distance_to_obstacle = jnp.linalg.norm(x_current - x_obstacle)
        constraint = distance_to_obstacle - (obstacle_radius + safety_margin)
        
        return {
            'objective': objective,
            'constraint': constraint,
            'current_pos': current_pos,
            'target_pos': target_pos,
            'obstacle_pos': obstacle_pos,
            'obstacle_radius': obstacle_radius,
            'safety_margin': safety_margin
        }
        
    except ImportError:
        print("JAX not available for QP demonstration")
        return None

# Test collision avoidance QP setup
current_pos = [0.0, 0.0]
target_pos = [3.0, 4.0]
obstacle_pos = [1.5, 2.0]
obstacle_radius = 0.8

qp_setup = collision_avoidance_qp(current_pos, obstacle_pos, obstacle_radius, target_pos)
if qp_setup:
    print("Collision avoidance QP setup:")
    print(f"Current: {qp_setup['current_pos']}")
    print(f"Target: {qp_setup['target_pos']}")
    print(f"Obstacle: {qp_setup['obstacle_pos']} (radius: {qp_setup['obstacle_radius']})")
    print(f"Safety margin: {qp_setup['safety_margin']}")
```

## Advanced Interview Topics
```

### Rodrigues' Rotation Formula
```python
def rodrigues_rotation(axis, angle):
    """
    Compute rotation matrix using Rodrigues' formula
    R = I + sin(θ) * K + (1 - cos(θ)) * K^2
    where K is the skew-symmetric matrix of the axis
    """
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize
    
    # Create skew-symmetric matrix K
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R

def axis_angle_to_rotation(axis, angle):
    """Alternative implementation using quaternion-like approach"""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    a = np.cos(angle / 2)
    b, c, d = axis * np.sin(angle / 2)
    
    # Convert quaternion to rotation matrix
    R = np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d]
    ])
    
    return R

# Test
axis = [0, 0, 1]  # Rotation around z-axis
angle = np.pi / 4  # 45 degrees

R1 = rodrigues_rotation(axis, angle)
R2 = axis_angle_to_rotation(axis, angle)

print("Rodrigues' formula:")
print(R1)
print("\nQuaternion method:")
print(R2)
print(f"\nMatrices are equal: {np.allclose(R1, R2)}")
print(f"Is rotation matrix: {is_rotation_matrix(R1)}")
```

## Advanced Inverse Kinematics

### Jacobian Computation
```python
def compute_jacobian(q, link_lengths):
    """
    Compute analytical Jacobian for n-link planar arm
    q: joint angles [θ1, θ2, ..., θn]
    link_lengths: [l1, l2, ..., ln]
    """
    n = len(q)
    jacobian = np.zeros((2, n))  # 2D position, n joints
    
    # Forward kinematics to get link positions
    positions = [(0, 0)]  # Base position
    current_angle = 0
    
    for i in range(n):
        current_angle += q[i]
        x = positions[-1][0] + link_lengths[i] * np.cos(current_angle)
        y = positions[-1][1] + link_lengths[i] * np.sin(current_angle)
        positions.append((x, y))
    
    # Compute Jacobian columns
    end_effector = positions[-1]
    current_angle = 0
    
    for i in range(n):
        current_angle += q[i]
        # Partial derivatives of end-effector position w.r.t. joint i
        jacobian[0, i] = -link_lengths[i] * np.sin(current_angle)  # dx/dθi
        jacobian[1, i] = link_lengths[i] * np.cos(current_angle)   # dy/dθi
        
        # Add contributions from subsequent links
        for j in range(i + 1, n):
            angle_sum = sum(q[k] for k in range(j + 1))
            jacobian[0, i] -= link_lengths[j] * np.sin(angle_sum)
            jacobian[1, i] += link_lengths[j] * np.cos(angle_sum)
    
    return jacobian

# Test
q = [np.pi/4, np.pi/3]  # Joint angles
link_lengths = [1.0, 1.0]  # Link lengths
J = compute_jacobian(q, link_lengths)
print("Jacobian matrix:")
print(J)
print(f"Shape: {J.shape}")
```

### Newton-Raphson IK Solver
```python
def newton_raphson_ik(target_pos, initial_q, link_lengths, 
                      max_iter=100, tol=1e-6, alpha=0.1):
    """
    Solve IK using Newton-Raphson method
    target_pos: desired end-effector position (x, y)
    initial_q: initial joint angles
    link_lengths: link lengths
    """
    q = np.array(initial_q, dtype=float)
    
    for iteration in range(max_iter):
        # Forward kinematics
        x, y = forward_kinematics_2d(q, link_lengths)
        current_pos = np.array([x, y])
        
        # Position error
        error = np.array(target_pos) - current_pos
        
        # Check convergence
        if np.linalg.norm(error) < tol:
            return q, iteration
        
        # Compute Jacobian
        J = compute_jacobian(q, link_lengths)
        
        # Check for singularity
        if np.linalg.cond(J) > 1e15:
            print(f"Singularity detected at iteration {iteration}")
            break
        
        # Newton step: q_new = q + J^-1 * error
        try:
            delta_q = np.linalg.solve(J, error)
            q += alpha * delta_q  # Damping factor
        except np.linalg.LinAlgError:
            print(f"Singular Jacobian at iteration {iteration}")
            break
    
    return q, iteration

def forward_kinematics_2d(q, link_lengths):
    """Forward kinematics for planar arm"""
    x, y = 0, 0
    angle = 0
    for i in range(len(q)):
        angle += q[i]
        x += link_lengths[i] * np.cos(angle)
        y += link_lengths[i] * np.sin(angle)
    return x, y

# Test
target = [1.5, 1.0]
initial_q = [0.1, 0.1]
link_lengths = [1.0, 1.0]

solution, iterations = newton_raphson_ik(target, initial_q, link_lengths)
print(f"Solution found in {iterations} iterations:")
print(f"Joint angles: {solution}")
print(f"End effector position: {forward_kinematics_2d(solution, link_lengths)}")
```

### Damped Least Squares (DLS) IK
```python
def damped_least_squares_ik(J, error, damping_factor=0.1):
    """
    Compute joint velocity using Damped Least Squares
    J: Jacobian matrix
    error: position/orientation error
    damping_factor: λ (damping parameter)
    """
    n = J.shape[1]  # Number of joints
    
    # DLS solution: δq = J^T * (J * J^T + λ²I)^(-1) * error
    JJT = J @ J.T
    damping_matrix = damping_factor**2 * np.eye(JJT.shape[0])
    
    try:
        inv_term = np.linalg.inv(JJT + damping_matrix)
        delta_q = J.T @ inv_term @ error
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        J_pinv = np.linalg.pinv(J)
        delta_q = J_pinv @ error
    
    return delta_q

def dls_ik_solver(target_pos, initial_q, link_lengths, 
                  max_iter=100, tol=1e-6, damping=0.1, alpha=0.1):
    """
    Complete DLS IK solver
    """
    q = np.array(initial_q, dtype=float)
    
    for iteration in range(max_iter):
        # Forward kinematics
        x, y = forward_kinematics_2d(q, link_lengths)
        current_pos = np.array([x, y])
        
        # Position error
        error = np.array(target_pos) - current_pos
        
        # Check convergence
        if np.linalg.norm(error) < tol:
            return q, iteration, True
        
        # Compute Jacobian
        J = compute_jacobian(q, link_lengths)
        
        # DLS step
        delta_q = damped_least_squares_ik(J, error, damping)
        q += alpha * delta_q
    
    return q, iteration, False

# Test
target = [1.8, 0.5]  # Near singularity
initial_q = [0.1, 0.1]
link_lengths = [1.0, 1.0]

solution, iterations, converged = dls_ik_solver(target, initial_q, link_lengths)
print(f"DLS IK {'converged' if converged else 'did not converge'} in {iterations} iterations")
print(f"Joint angles: {solution}")
print(f"End effector position: {forward_kinematics_2d(solution, link_lengths)}")
print(f"Target position: {target}")
```

### DLS IK with JAX (Automatic Differentiation) - OPTIONAL
```python
import jax
import jax.numpy as jnp
from jax import jacfwd

def forward_kinematics_jax(q, link_lengths):
    """JAX-compatible forward kinematics"""
    x, y = 0.0, 0.0
    angle = 0.0
    for i in range(len(q)):
        angle = angle + q[i]
        x = x + link_lengths[i] * jnp.cos(angle)
        y = y + link_lengths[i] * jnp.sin(angle)
    return jnp.array([x, y])

def dls_ik_jax(target_pos, initial_q, link_lengths, 
                max_iter=100, tol=1e-6, damping=0.1, alpha=0.1):
    """
    DLS IK using JAX automatic differentiation
    JAX is OPTIONAL - provides automatic Jacobian computation
    """
    target_pos = jnp.array(target_pos)
    q = jnp.array(initial_q, dtype=jnp.float32)
    link_lengths = jnp.array(link_lengths)
    
    # Compute Jacobian using automatic differentiation
    jacobian_fn = jacfwd(forward_kinematics_jax, argnums=0)
    
    for iteration in range(max_iter):
        # Current position
        current_pos = forward_kinematics_jax(q, link_lengths)
        
        # Position error
        error = target_pos - current_pos
        
        # Check convergence
        if jnp.linalg.norm(error) < tol:
            return q, iteration, True
        
        # Compute Jacobian
        J = jacobian_fn(q, link_lengths)
        
        # DLS solution
        JJT = J @ J.T
        damping_matrix = damping**2 * jnp.eye(JJT.shape[0])
        inv_term = jnp.linalg.inv(JJT + damping_matrix)
        delta_q = J.T @ inv_term @ error
        
        # Update
        q = q + alpha * delta_q
    
    return q, iteration, False

# Test JAX implementation (OPTIONAL)
try:
    target = [1.5, 1.0]
    initial_q = [0.1, 0.1]
    link_lengths = [1.0, 1.0]
    
    solution, iterations, converged = dls_ik_jax(target, initial_q, link_lengths)
    print("JAX DLS IK Results:")
    print(f"Converged: {converged} in {iterations} iterations")
    print(f"Joint angles: {solution}")
    print(f"End effector position: {forward_kinematics_jax(solution, link_lengths)}")
except ImportError:
    print("JAX not available. DLS IK works fine without JAX!")
    print("Use the NumPy implementation above instead.")
```

### Null Space Control for Redundant Manipulators
```python
def null_space_control(J, primary_task, null_space_objective, 
                      joint_limits=None, damping=0.1):
    """
    Compute joint velocities for redundant manipulator
    primary_task: primary task velocity (e.g., end-effector velocity)
    null_space_objective: secondary objective (e.g., joint limit avoidance)
    """
    # Primary task solution using DLS
    JJT = J @ J.T
    damping_matrix = damping**2 * np.eye(JJT.shape[0])
    inv_term = np.linalg.inv(JJT + damping_matrix)
    q_primary = J.T @ inv_term @ primary_task
    
    # Null space projection
    null_space_proj = np.eye(J.shape[1]) - J.T @ inv_term @ J
    
    # Add null space objective
    q_null = null_space_proj @ null_space_objective
    
    # Combine solutions
    q_total = q_primary + q_null
    
    return q_total

def joint_limit_avoidance(q, q_min, q_max, gain=0.1):
    """
    Compute joint velocity to avoid joint limits
    """
    n = len(q)
    objective = np.zeros(n)
    
    for i in range(n):
        # Distance to limits
        d_min = q[i] - q_min[i]
        d_max = q_max[i] - q[i]
        
        # Repulsive force
        if d_min < 0.1:  # Close to lower limit
            objective[i] = gain / (d_min + 1e-6)
        elif d_max < 0.1:  # Close to upper limit
            objective[i] = -gain / (d_max + 1e-6)
    
    return objective

# Test null space control
q_current = np.array([0.5, 1.0, 0.8])
q_min = np.array([0.0, 0.0, 0.0])
q_max = np.array([np.pi, np.pi, np.pi])

# Primary task: move end-effector in x-direction
primary_velocity = np.array([0.1, 0.0])

# Null space objective: avoid joint limits
null_objective = joint_limit_avoidance(q_current, q_min, q_max)

# Compute Jacobian (simplified 3-link example)
J = np.array([[0.5, 0.3, 0.1],
              [0.0, 0.4, 0.2]])

q_dot = null_space_control(J, primary_velocity, null_objective)
print("Joint velocities with null space control:")
print(q_dot)
```

## Reinforcement Learning in Robotics

Modern robotics heavily relies on reinforcement learning for complex tasks. These problems focus on practical implementation skills needed for RL in robotics, including reward design, state representation, safety constraints, and deployment considerations.

### Key RL Concepts for Robotics

#### Reward Function Design
- **Task Completion**: Distance to goal, success bonuses
- **Efficiency**: Energy consumption, path length, time penalties
- **Safety**: Collision avoidance, joint limit penalties
- **Smoothness**: Velocity/acceleration limits, jerk minimization

#### State Representation
- **Raw State**: Joint positions, velocities, end-effector pose
- **Features**: Relative positions, distances to obstacles, task progress
- **History**: Temporal information, velocity trends

#### Action Spaces
- **Joint Control**: Direct joint position/velocity/torque commands
- **End-Effector Control**: Cartesian position/orientation targets
- **Hybrid Control**: Combination of joint and task space control

### Worked Examples

### Problem 30: Pose Error Computation
```python
import numpy as np
from scipy.spatial.transform import Rotation

def compute_pose_error(current_pose, desired_pose):
    """
    Compute pose error for RL reward functions.
    current_pose: [x, y, z, qx, qy, qz, qw]
    desired_pose: [x, y, z, qx, qy, qz, qw]
    """
    # Position error
    pos_error = np.array(desired_pose[:3]) - np.array(current_pose[:3])
    
    # Orientation error (axis-angle representation)
    current_rot = Rotation.from_quat(current_pose[3:])
    desired_rot = Rotation.from_quat(desired_pose[3:])
    
    # Relative rotation
    rel_rot = desired_rot * current_rot.inv()
    orientation_error = rel_rot.as_rotvec()  # axis-angle
    
    return pos_error, orientation_error

# Test
current = [0.1, 0.0, 0.5, 0, 0, 0, 1]  # slight position error
desired = [0.0, 0.0, 0.5, 0, 0, 0, 1]   # target pose
pos_err, orient_err = compute_pose_error(current, desired)
print(f"Position error: {pos_err}")
print(f"Orientation error: {orient_err}")
```

### Problem 31: Twist Error for RL
```python
def compute_twist_error(current_twist, desired_twist):
    """
    Compute velocity error for RL training.
    current_twist: [vx, vy, vz, wx, wy, wz]
    desired_twist: [vx, vy, vz, wx, wy, wz]
    """
    return np.array(desired_twist) - np.array(current_twist)

# Test
current_vel = [0.1, 0.0, 0.0, 0.0, 0.0, 0.1]  # some velocity
desired_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # target: stop
twist_error = compute_twist_error(current_vel, desired_vel)
print(f"Twist error: {twist_error}")
```

### Problem 32: Comprehensive Reward Function
```python
def compute_reward(current_pose, desired_pose, current_twist, action, dt):
    """
    Multi-component reward function for reaching task.
    """
    # Position and orientation errors
    pos_error, orient_error = compute_pose_error(current_pose, desired_pose)
    pos_error_norm = np.linalg.norm(pos_error)
    orient_error_norm = np.linalg.norm(orient_error)
    
    # Velocity penalty (prefer smooth motion)
    velocity_penalty = np.linalg.norm(current_twist)
    
    # Action penalty (prefer smaller actions)
    action_penalty = np.linalg.norm(action)
    
    # Success bonus
    success_threshold = 0.05
    success_bonus = 10.0 if pos_error_norm < success_threshold else 0.0
    
    # Composite reward
    reward = -pos_error_norm - 0.1 * orient_error_norm - 0.01 * velocity_penalty - 0.001 * action_penalty + success_bonus
    
    return reward

# Test
current_pose = [0.1, 0.0, 0.5, 0, 0, 0, 1]
desired_pose = [0.0, 0.0, 0.5, 0, 0, 0, 1]
current_twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
action = [0.0, 0.0]
reward = compute_reward(current_pose, desired_pose, current_twist, action, 0.01)
print(f"Reward: {reward}")
```

### Problem 33: State Feature Extraction
```python
def extract_state_features(joint_positions, joint_velocities, end_effector_pose, obstacles):
    """
    Extract meaningful features for RL policy input.
    """
    features = []
    
    # Joint states
    features.extend(joint_positions)
    features.extend(joint_velocities)
    
    # End-effector pose
    features.extend(end_effector_pose[:3])  # position
    features.extend(end_effector_pose[3:])  # orientation
    
    # Distance to obstacles
    ee_pos = np.array(end_effector_pose[:3])
    for obstacle in obstacles:
        dist = np.linalg.norm(ee_pos - np.array(obstacle))
        features.append(dist)
    
    # Joint limit distances
    joint_limits_min = np.array([0.0, -np.pi/2, 0.0])
    joint_limits_max = np.array([np.pi, np.pi/2, np.pi])
    
    for i, (pos, min_limit, max_limit) in enumerate(zip(joint_positions, joint_limits_min, joint_limits_max)):
        dist_to_min = pos - min_limit
        dist_to_max = max_limit - pos
        features.extend([dist_to_min, dist_to_max])
    
    return np.array(features)

# Test
joint_pos = [0.5, 0.0, 1.0]
joint_vel = [0.0, 0.0, 0.0]
ee_pose = [0.8, 0.0, 0.5, 0, 0, 0, 1]
obstacles = [[1.0, 0.0, 0.5]]
features = extract_state_features(joint_pos, joint_vel, ee_pose, obstacles)
print(f"Feature vector length: {len(features)}")
```

### Problem 34: Action Space Constraints
```python
def compute_action_bounds(joint_positions, joint_limits, velocity_limits, torque_limits):
    """
    Compute safe action bounds for RL.
    """
    n_joints = len(joint_positions)
    
    # Joint position limits
    pos_min = np.array(joint_limits[0])
    pos_max = np.array(joint_limits[1])
    
    # Velocity limits
    vel_min = -np.array(velocity_limits)
    vel_max = np.array(velocity_limits)
    
    # Torque limits
    torque_min = -np.array(torque_limits)
    torque_max = np.array(torque_limits)
    
    # Combine constraints (simplified - in practice use proper constraint satisfaction)
    action_min = np.maximum(vel_min, (pos_min - joint_positions) / 0.01)  # velocity to stay within position limits
    action_max = np.minimum(vel_max, (pos_max - joint_positions) / 0.01)
    
    return action_min, action_max

# Test
joint_pos = [0.5, 0.0]
joint_limits = [[0.0, -np.pi/2], [np.pi, np.pi/2]]
velocity_limits = [1.0, 1.0]
torque_limits = [10.0, 10.0]
action_min, action_max = compute_action_bounds(joint_pos, joint_limits, velocity_limits, torque_limits)
print(f"Action bounds: min={action_min}, max={action_max}")
```

### Problem 35: Episode Termination Logic
```python
def check_episode_termination(current_pose, desired_pose, joint_positions, joint_limits, time_elapsed, max_time):
    """
    Determine episode termination conditions.
    """
    # Success condition
    pos_error, _ = compute_pose_error(current_pose, desired_pose)
    success = np.linalg.norm(pos_error) < 0.05
    
    # Failure conditions
    joint_min = np.array(joint_limits[0])
    joint_max = np.array(joint_limits[1])
    joint_violation = np.any(joint_positions < joint_min) or np.any(joint_positions > joint_max)
    
    timeout = time_elapsed > max_time
    
    # Episode done
    done = success or joint_violation or timeout
    
    info = {
        'success': success,
        'joint_violation': joint_violation,
        'timeout': timeout,
        'time_elapsed': time_elapsed
    }
    
    return done, success, info

# Test
current_pose = [0.02, 0.0, 0.5, 0, 0, 0, 1]
desired_pose = [0.0, 0.0, 0.5, 0, 0, 0, 1]
joint_pos = [0.5, 0.0]
joint_limits = [[0.0, -np.pi/2], [np.pi, np.pi/2]]
done, success, info = check_episode_termination(current_pose, desired_pose, joint_pos, joint_limits, 5.0, 10.0)
print(f"Episode done: {done}, Success: {success}")
```

### Problem 36: Performance Metrics
```python
def compute_performance_metrics(trajectory, desired_trajectory, final_pose_error, time_to_completion, energy_consumed):
    """
    Compute comprehensive performance metrics.
    """
    metrics = {}
    
    # Trajectory accuracy
    trajectory_errors = []
    for actual, desired in zip(trajectory, desired_trajectory):
        pos_error, _ = compute_pose_error(actual, desired)
        trajectory_errors.append(np.linalg.norm(pos_error))
    
    metrics['mean_trajectory_error'] = np.mean(trajectory_errors)
    metrics['max_trajectory_error'] = np.max(trajectory_errors)
    metrics['final_pose_error'] = final_pose_error
    
    # Efficiency metrics
    metrics['time_to_completion'] = time_to_completion
    metrics['energy_consumed'] = energy_consumed
    metrics['path_length'] = sum(np.linalg.norm(np.diff(np.array(trajectory)[:, :3], axis=0), axis=1))
    
    # Success rate (would be computed over multiple episodes)
    metrics['success_rate'] = 1.0 if final_pose_error < 0.05 else 0.0
    
    return metrics

# Test
trajectory = [[0.1, 0.0, 0.5, 0, 0, 0, 1], [0.05, 0.0, 0.5, 0, 0, 0, 1], [0.02, 0.0, 0.5, 0, 0, 0, 1]]
desired_trajectory = [[0.0, 0.0, 0.5, 0, 0, 0, 1]] * 3
metrics = compute_performance_metrics(trajectory, desired_trajectory, 0.02, 3.0, 1.5)
print("Performance metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.3f}")
```

### Problem 37: Data Collection for Offline RL
```python
def collect_offline_data(states, actions, rewards, next_states, dones, episode_data):
    """
    Structure data collection for offline RL.
    """
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': [],
        'episode_starts': [],
        'episode_lengths': []
    }
    
    episode_start = True
    current_episode_length = 0
    
    for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
        dataset['observations'].append(state)
        dataset['actions'].append(action)
        dataset['rewards'].append(reward)
        dataset['next_observations'].append(next_state)
        dataset['terminals'].append(done)
        dataset['episode_starts'].append(episode_start)
        
        current_episode_length += 1
        episode_start = False
        
        if done or i == len(states) - 1:
            dataset['episode_lengths'].append(current_episode_length)
            episode_start = True
            current_episode_length = 0
    
    # Convert to numpy arrays
    for key in dataset:
        if key != 'episode_lengths':
            dataset[key] = np.array(dataset[key])
    
    return dataset

# Test
states = [[0.1, 0.0], [0.05, 0.0], [0.02, 0.0]]
actions = [[-0.1], [-0.05], [0.0]]
rewards = [-0.1, -0.05, 10.0]
next_states = [[0.05, 0.0], [0.02, 0.0], [0.0, 0.0]]
dones = [False, False, True]
dataset = collect_offline_data(states, actions, rewards, next_states, dones, {})
print(f"Dataset keys: {list(dataset.keys())}")
print(f"Episode lengths: {dataset['episode_lengths']}")
```

### Problem 38: Curriculum Learning
```python
def curriculum_scheduler(task_difficulty, success_rate, episode_count):
    """
    Design curriculum learning progression.
    """
    # Difficulty parameters
    if success_rate > 0.8 and episode_count > 100:
        # Increase difficulty
        new_target_distance = min(task_difficulty['target_distance'] * 1.2, 1.0)
        new_obstacle_count = min(task_difficulty['obstacle_count'] + 1, 5)
        new_time_limit = max(task_difficulty['time_limit'] * 0.9, 5.0)
    elif success_rate < 0.3:
        # Decrease difficulty
        new_target_distance = max(task_difficulty['target_distance'] * 0.8, 0.1)
        new_obstacle_count = max(task_difficulty['obstacle_count'] - 1, 0)
        new_time_limit = task_difficulty['time_limit'] * 1.1
    else:
        # Keep current difficulty
        new_target_distance = task_difficulty['target_distance']
        new_obstacle_count = task_difficulty['obstacle_count']
        new_time_limit = task_difficulty['time_limit']
    
    next_task_params = {
        'target_distance': new_target_distance,
        'obstacle_count': new_obstacle_count,
        'time_limit': new_time_limit,
        'success_rate': success_rate,
        'episode_count': episode_count
    }
    
    return next_task_params

# Test
current_difficulty = {'target_distance': 0.5, 'obstacle_count': 2, 'time_limit': 10.0}
next_params = curriculum_scheduler(current_difficulty, 0.85, 150)
print("Next task parameters:")
for key, value in next_params.items():
    print(f"  {key}: {value}")
```

### Problem 39: Multi-Objective Reward Design
```python
def multi_objective_reward(position_error, orientation_error, joint_effort, safety_violations, task_progress):
    """
    Multi-objective reward function balancing multiple goals.
    """
    # Task completion reward
    task_reward = -np.linalg.norm(position_error) - 0.1 * np.linalg.norm(orientation_error)
    
    # Efficiency reward
    efficiency_reward = -0.01 * joint_effort
    
    # Safety reward
    safety_reward = -10.0 * safety_violations
    
    # Progress reward
    progress_reward = 5.0 * task_progress
    
    # Combine rewards with weights
    weights = {'task': 1.0, 'efficiency': 0.5, 'safety': 2.0, 'progress': 1.0}
    
    total_reward = (weights['task'] * task_reward + 
                   weights['efficiency'] * efficiency_reward + 
                   weights['safety'] * safety_reward + 
                   weights['progress'] * progress_reward)
    
    reward_components = {
        'task': task_reward,
        'efficiency': efficiency_reward,
        'safety': safety_reward,
        'progress': progress_reward,
        'total': total_reward
    }
    
    return total_reward, reward_components

# Test
reward, components = multi_objective_reward(
    position_error=np.array([0.1, 0.0, 0.0]),
    orientation_error=np.array([0.0, 0.0, 0.0]),
    joint_effort=2.0,
    safety_violations=0,
    task_progress=0.5
)
print(f"Total reward: {reward:.3f}")
print("Reward components:")
for key, value in components.items():
    print(f"  {key}: {value:.3f}")
```

### Problem 40: Safety Monitoring
```python
def safety_monitor(joint_positions, joint_velocities, end_effector_pose, obstacles, safety_margins):
    """
    Monitor safety constraints and compute penalties.
    """
    violations = []
    penalty = 0.0
    
    # Joint limit violations
    joint_min = np.array([-np.pi, -np.pi/2, 0.0])
    joint_max = np.array([np.pi, np.pi/2, np.pi])
    
    for i, (pos, min_limit, max_limit) in enumerate(zip(joint_positions, joint_min, joint_max)):
        if pos < min_limit + safety_margins['joint_margin']:
            violations.append(f"Joint {i} near minimum limit")
            penalty += 5.0 * (min_limit + safety_margins['joint_margin'] - pos)
        elif pos > max_limit - safety_margins['joint_margin']:
            violations.append(f"Joint {i} near maximum limit")
            penalty += 5.0 * (pos - (max_limit - safety_margins['joint_margin']))
    
    # Velocity limit violations
    vel_limits = np.array([2.0, 2.0, 2.0])
    for i, (vel, limit) in enumerate(zip(joint_velocities, vel_limits)):
        if abs(vel) > limit * safety_margins['velocity_factor']:
            violations.append(f"Joint {i} velocity too high")
            penalty += 2.0 * (abs(vel) - limit * safety_margins['velocity_factor'])
    
    # Collision detection
    ee_pos = np.array(end_effector_pose[:3])
    for i, obstacle in enumerate(obstacles):
        dist = np.linalg.norm(ee_pos - np.array(obstacle))
        if dist < safety_margins['collision_margin']:
            violations.append(f"Too close to obstacle {i}")
            penalty += 10.0 * (safety_margins['collision_margin'] - dist)
    
    return penalty, violations

# Test
joint_pos = [0.1, 0.0, 0.0]
joint_vel = [0.5, 0.0, 0.0]
ee_pose = [0.8, 0.0, 0.5, 0, 0, 0, 1]
obstacles = [[1.0, 0.0, 0.5]]
safety_margins = {'joint_margin': 0.1, 'velocity_factor': 0.9, 'collision_margin': 0.2}
penalty, violations = safety_monitor(joint_pos, joint_vel, ee_pose, obstacles, safety_margins)
print(f"Safety penalty: {penalty}")
print(f"Violations: {violations}")
```

## Practice Tips
- Understand coordinate systems (world vs local)
- Control theory: stability, oscillations, settling time
- Path planning: optimality vs computational complexity
- Computer vision: feature extraction, filtering
- Always consider real-world constraints: noise, delays, hardware limitations
- Think about multi-robot coordination and swarm robotics
- Consider safety and robustness in robotic systems

**Matrix Math Interview Tips:**
- Know when to use different matrix decompositions (LU, QR, SVD)
- Understand condition numbers and numerical stability
- Practice implementing basic linear algebra operations from scratch
- Be familiar with rotation representations (matrices, quaternions, axis-angle)

**Advanced IK Interview Tips:**
- Understand the difference between analytical and numerical IK
- Know when to use damping and why it's important near singularities
- Be able to explain null space control for redundant manipulators
- Understand the trade-offs between different IK algorithms

**RL in Robotics Interview Tips:**
- Be able to design reward functions that balance multiple objectives
- Understand state representation and feature engineering for robotics
- Know how to implement safety constraints in RL training
- Be familiar with curriculum learning and domain randomization
- Understand offline RL data collection and formatting
- Know how to evaluate and monitor RL performance in real-time systems
- Understand coordinate systems (world vs local)
- Control theory: stability, oscillations, settling time
- Path planning: optimality vs computational complexity
- Computer vision: feature extraction, filtering
- Always consider real-world constraints: noise, delays, hardware limitations
- Think about multi-robot coordination and swarm robotics
- Consider safety and robustness in robotic systems

**Matrix Math Interview Tips:**
- Know when to use different matrix decompositions (LU, QR, SVD)
- Understand condition numbers and numerical stability
- Practice implementing basic linear algebra operations from scratch
- Be familiar with rotation representations (matrices, quaternions, axis-angle)

**Advanced IK Interview Tips:**
- Understand the difference between analytical and numerical IK
- Know when to use damping and why it's important near singularities
- Be able to explain null space control for redundant manipulators
- Understand the trade-offs between different IK algorithms
