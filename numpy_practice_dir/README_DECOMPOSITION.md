# LU, QR, and Cholesky Decompositions — From First Principles

This README is a teaching-first tutorial on **LU**, **QR**, and **Cholesky** decompositions. It defines all variables, derives each method from scratch, gives stable algorithms, and shows how to use each factorization for solving systems and least-squares problems.

---

## 0) Notation & Preliminaries
Let
- $A \in \mathbb{R}^{m\times n}$: input matrix (square when stated).
- $I$: identity matrix; $T$: transpose; $\ast$: conjugate transpose in the complex case.
- $\|\cdot\|_2$: spectral norm; $\|\cdot\|_F$: Frobenius norm.
- A matrix is **upper triangular** if entries below the diagonal are zero, **lower triangular** if entries above are zero.

We will repeatedly use the normal equations identity and orthogonality: $Q^TQ=I$ for orthogonal $Q$.

---

## 1) LU Factorization (with Partial Pivoting)

### Goal
For a **square** matrix $A\in\mathbb{R}^{n\times n}$, find
$$
P\,A \,=\, L\,U,
$$
Where:
- $P\in\mathbb{R}^{n\times n}$ is a **permutation** matrix encoding row swaps (partial pivoting),
- $L\in\mathbb{R}^{n\times n}$ is **unit lower triangular** (ones on the diagonal),
- $U\in\mathbb{R}^{n\times n}$ is **upper triangular**.

**Why it matters:** to solve $A x = b$ efficiently via two triangular solves:
1) $L y = P b$ (forward substitution), 2) $U x = y$ (back substitution).

### Derivation via Gaussian Elimination
Gaussian elimination zeros subdiagonal entries column by column. At step $k$, we eliminate $A_{ik}$ for $i>k$ using multipliers
$$
L_{ik} \,=\, \frac{A^{(k)}_{ik}}{A^{(k)}_{kk}},\quad i=k+1,\dots,n,
$$
then update the trailing submatrix
$$
A^{(k+1)}_{ij} \,=\, A^{(k)}_{ij} - L_{ik}\,A^{(k)}_{kj},\quad j=k,\dots,n.
$$
Collecting all unit-lower “Gauss transforms” yields $A = L U$. To avoid division by tiny pivots and reduce growth, we first swap the maximal-magnitude pivot into position $k$—this is **partial pivoting**, encoded by $P$ in $P A = L U$.

### Doolittle Algorithm (stable form)
For $k=1,\dots,n$:
1. **Pivot:** find $p\ge k$ maximizing $|A^{(k)}_{p,k}|$; swap rows $k\leftrightarrow p$ in $A$ and record in $P$; swap the first $k-1$ columns of $L$ accordingly.
2. **Multipliers:** $L_{ik} = A^{(k)}_{ik}/A^{(k)}_{kk}$ for $i=k+1,\dots,n$; set $L_{kk}=1$.
3. **Update row of $U$:** the $k$‑th row of $U$ becomes the updated row $A^{(k)}_{k,\,k:n}$.
4. **Schur update:** $A^{(k+1)}_{i,\,k:n} \leftarrow A^{(k)}_{i,\,k:n} - L_{ik} A^{(k)}_{k,\,k:n}$ for $i>k$.

### Using LU
**Solve:** $A x = b \Rightarrow$ (i) $y = (L)^{-1} (P b)$, (ii) $x = U^{-1} y$.

**Determinant:** $\det(A)=\det(P)\prod_{i=1}^n U_{ii}$.

**Cost:** $\tfrac{2}{3}n^3$ flops to factor; each solve is $\mathcal{O}(n^2)$.

**Stability:** LU with **partial pivoting** is backward stable in practice for most problems; pathological growth factors exist but are rare.

**Sanity checks:** $\|PA - L U\|_F/\|A\|_F$ small; $U$ upper triangular; $L$ unit lower.

---

## 2) QR Factorization (Householder or Modified Gram–Schmidt)

### Goal
For $A\in\mathbb{R}^{m\times n}$ with $m\ge n$, find
$$
A \,=\, Q\,R,
$$
Where:
- $Q\in\mathbb{R}^{m\times m}$ is **orthogonal**: $Q^T Q = I$,
- $R\in\mathbb{R}^{m\times n}$ has an **upper‑triangular** leading $n\times n$ block (often we store only $R_{1:n,1:n}$ and $Q_{:,1:n}$).

**Why it matters:** numerically robust **least‑squares** solver
$$
\min_x \|A x - b\|_2^2 \quad\Rightarrow\quad A=QR,\ \ Q^T A = R,\ \ Q^T b = y \Rightarrow R_{1:n,1:n} x = y_{1:n}.
$$

### From first principles
**(a) Classical/Modified Gram–Schmidt** (conceptual): orthonormalize columns $a_j$ of $A$ to get $q_j$ and triangular $R$ via
$$
\begin{aligned}
&r_{ij} = q_i^T a_j,\quad \tilde a_j = a_j - \sum_{i=1}^{j-1} r_{ij} q_i,\\
&r_{jj} = \|\tilde a_j\|_2,\quad q_j = \tilde a_j/r_{jj}.
\end{aligned}
$$
Classical GS can be unstable for nearly dependent columns; **Modified GS** is better but still weaker than Householder in finite precision.

**(b) Householder reflections** (preferred): a reflector
$$
H = I - 2\,\frac{v v^T}{v^T v}
$$
maps a vector to a multiple of $e_1$ when $v = x \pm \|x\|_2 e_1$ (sign chosen to avoid cancellation). Apply a sequence $H_1,\dots,H_n$ to zero subdiagonals column by column:
$$
R = H_n \cdots H_1 A,\qquad Q = H_1^T\cdots H_n^T.
$$

### Using QR
- **Least squares:** compute $y=Q^T b$, solve $R_{1:n,1:n} x = y_{1:n}$.
- **Rank revealing:** use **pivoted QR** if $A$ may be rank‑deficient.

**Cost:** about $\tfrac{2}{3} m n^2$ for tall $m\gg n$ (square case: $\tfrac{2}{3}n^3$).

**Stability:** Householder QR is backward stable; Modified GS is acceptable; Classical GS is not.

**Sanity checks:** $\|A - Q R\|_F/\|A\|_F$ small; $\|Q^T Q - I\|_F$ small; $R$ upper triangular.

---

## 3) Cholesky Factorization (SPD Matrices)

### Goal
For a **symmetric positive definite** (SPD) matrix $A\in\mathbb{R}^{n\times n}$, find
$$
A \,=\, L\,L^T,
$$
Where $L$ is **lower triangular with positive diagonal**.

**Why it matters:** fastest, most stable method to solve SPD systems; also useful for log‑determinants and sampling Gaussians.

### Derivation (scalar recurrences)
For $i=1,\dots,n$:
$$
L_{ii} \,=\, \sqrt{\,A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2\,},\qquad
L_{ji} \,=\, \frac{A_{ji} - \sum_{k=1}^{i-1} L_{jk} L_{ik}}{L_{ii}},\quad j=i+1,\dots,n.
$$
(Upper form computes $A=U^T U$.) If any $L_{ii}\le 0$ arises (beyond round‑off), $A$ is not SPD or is ill‑conditioned.

### Using Cholesky
Solve $A x = b$ by (i) $L y = b$ (forward), (ii) $L^T x = y$ (backward).  
**Log‑det:** $\log\det(A) = 2\sum_{i=1}^n \log L_{ii}$.

**Cost:** $\tfrac{1}{3}n^3$ flops to factor; each solve is $\mathcal{O}(n^2)$.

**Sanity checks:** $\|A - L L^T\|_F/\|A\|_F$ small; all $L_{ii}>0$.

---

## 4) Choosing the Right Factorization
- **General square $A$:** LU with partial pivoting ($P A = L U$).
- **Least squares ($m\ge n$):** QR (Householder), or pivoted QR for rank‑deficient/ill‑conditioned cases.
- **SPD (or Hermitian PD):** Cholesky ($A=L L^T$) — fastest & stablest.

---

## 5) Algorithms (Minimal Pseudocode)

### LU with Partial Pivoting (Doolittle form)
```text
function [P,L,U] = lu_pp(A):
    n = size(A,1); P = I; L = I; U = A
    for k = 1..n-1:
        p = argmax_{i≥k} |U[i,k]|
        swap rows k,p in U and P; swap rows k,p in L[:,1:k-1]
        for i = k+1..n:
            L[i,k] = U[i,k] / U[k,k]
            U[i,k:n] -= L[i,k] * U[k,k:n]
    return P,L,U
```

### QR via Householder Reflections
```text
function [Q,R] = qr_householder(A):
    m,n = size(A); Q = I_m; R = A
    for k = 1..n:
        x = R[k:m, k]
        v = x + sign(x1)*||x||_2 * e1
        v = v / ||v||_2
        Hk = I_{m-k+1} - 2 v v^T
        R[k:m, k:n] = Hk * R[k:m, k:n]
        Q[:, k:m] = Q[:, k:m] * [I 0; 0 Hk]^T
    return Q, R
```

### Cholesky (Lower)
```text
function L = chol_lower(A):
    n = size(A,1); L = zeros(n,n)
    for i = 1..n:
        s = sum_{k=1}^{i-1} L[i,k]^2
        L[i,i] = sqrt(A[i,i] - s)
        for j = i+1..n:
            s = sum_{k=1}^{i-1} L[j,k]*L[i,k]
            L[j,i] = (A[j,i] - s)/L[i,i]
    return L
```

---

## 6) Worked Micro‑Examples (Formulas You Can Plug In)

### 6.1 Least Squares via QR
Given overdetermined $A\in\mathbb{R}^{m\times n}$ ($m>n$) and $b\in\mathbb{R}^m$:
$$
\min_x \|A x - b\|_2^2,\quad A = Q R,\quad y = Q^T b,\quad R_{1:n,1:n} x = y_{1:n}.
$$
This avoids forming the normal equations $A^T A x = A^T b$, which squares the condition number.

### 6.2 Solving SPD Systems via Cholesky
If $A$ is SPD and $A=L L^T$:
$$
L y = b \ (\text{forward}), \quad L^T x = y \ (\text{backward}).
$$
This is **half the flops** of LU on average and more stable.

### 6.3 Determinants
For triangular $U$ or $L$, the determinant is the product of diagonal entries. For LU with pivoting:
$$
\det(A) = \det(P)\prod_{i=1}^n U_{ii},\quad \det(P)\in\{+1,-1\}.
$$
For Cholesky: $\det(A) = (\prod_i L_{ii})^2$.

---

## 7) Stability & Diagnostics (What to Watch)
- **LU**: partial pivoting is the default; if you observe explosive growth in $U$, consider complete pivoting or switch to QR.
- **QR**: prefer Householder to classical Gram–Schmidt; Modified GS is acceptable for moderate conditioning.
- **Cholesky**: fails if $A$ is not SPD; for symmetric indefinite matrices, use **$LDL^T$ with pivoting**.

**Residual checks** after factorization:
$$
\frac{\|P A - L U\|_F}{\|A\|_F},\quad
\frac{\|A - Q R\|_F}{\|A\|_F},\quad
\frac{\|A - L L^T\|_F}{\|A\|_F}
\quad\text{(all should be small)}.
$$
Orthogonality check: $\|Q^T Q - I\|_F$ small. SPD check: $x^T A x>0$ for random $x\ne 0$.

---

## 8) (Optional) Linear Regression Example Template

As a concrete pattern you can reuse in derivations, linear regression fits
$$
\mathbf{y} = \mathbf{X} \, \boldsymbol{\beta} + \boldsymbol{\epsilon}
$$
Where:
- $\mathbf{y}\in\mathbb{R}^m$: observations.
- $\mathbf{X}\in\mathbb{R}^{m\times n}$: inputs (first column ones for intercept).
- $\boldsymbol{\beta}\in\mathbb{R}^n$: coefficients to learn.
- $\boldsymbol{\epsilon}\in\mathbb{R}^m$: noise.
- $m$: number of observations.

To solve for $\boldsymbol{\beta}$ stably, use **QR**:
$$
\min_{\boldsymbol{\beta}}\|\mathbf{X}\boldsymbol{\beta}-\mathbf{y}\|_2^2,\quad \mathbf{X}=Q R\Rightarrow R \boldsymbol{\beta}=Q^T\mathbf{y}.
$$
Avoid forming $(\mathbf{X}^T \mathbf{X})$ explicitly.

---

## 9) Complexity Summary (Square $n\times n$)
- **LU (PP):** $\tfrac{2}{3}n^3$ flops; solves $\mathcal{O}(n^2)$.
- **QR (Householder):** $\tfrac{2}{3}n^3$ flops.
- **Cholesky:** $\tfrac{1}{3}n^3$ flops; solves $\mathcal{O}(n^2)$.

---

## 10) Quick Reference — When to Use What
- **General square systems:** LU with partial pivoting.
- **Least squares / tall matrices / rank issues:** QR (Householder or pivoted QR).
- **SPD systems:** Cholesky.

---

If you want, I can add NumPy/PyTorch reference implementations and a tiny test harness that checks residuals and orthogonality for each factorization.

