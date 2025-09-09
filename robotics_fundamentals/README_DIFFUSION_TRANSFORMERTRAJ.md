# Diffusion Policies (DP) and Trajectory Transformers (TT)

## TL;DR
- **Diffusion Policies** learn a conditional **denoising** process over future action sequences; excellent for multimodal tasks and long horizons.
- **Trajectory Transformers** model full trajectories with autoregressive sequence models; plan by sampling high‑return sequences.

---

## 1) Diffusion policies from first principles
We wish to model an action sequence $\mathbf{a}_{0:H-1}\in\mathbb{R}^{H\times d_a}$ conditioned on context $c$ (images, proprioception, goals).

### Forward (noising) process
For $k=1,\dots,K$ with schedule $\{\beta_k\}$, define
$$
q\!\left(\mathbf{a}^{(k)}\mid\mathbf{a}^{(k-1)}\right)
= \mathcal{N}\!\left(\sqrt{1-\beta_k}\,\mathbf{a}^{(k-1)},\ \beta_k I\right),
$$
with $\mathbf{a}^{(0)}=\mathbf{a}$ and $\mathbf{a}^{(K)}$ approximately Gaussian.

### Reverse (denoising) model
Train a network $\epsilon_\theta(\mathbf{a}^{(k)}, k, c)$ to predict injected noise $\epsilon$ via
$$
\min_{\theta} \ \mathbb{E}\Big[\,\big\|\epsilon - \epsilon_\theta(\mathbf{a}^{(k)},k,c)\big\|_2^2\,\Big].
$$
At test time, start from $\mathbf{a}^{(K)}\sim\mathcal{N}(0,I)$ and iteratively denoise to $\mathbf{a}^{(0)}$; execute first $h\ll H$ actions, then recede the window.

**Conditioning**: time embeddings, goal tokens, cross‑attention over visual features.

---

## 2) Trajectory transformers from first principles
Model a trajectory $\tau=(s_0,a_0,r_0,\dots,s_T)$ as a token sequence and fit an autoregressive distribution
$$
p_\theta(\tau) = \prod_{t=0}^{T} p_\theta(x_t\mid x_{<t}),
$$
where each $x_t$ is a tokenized piece (state, action, return). Planning variants:
- **Return‑conditioned**: condition on a desired return‑to‑go token; sample actions via beam search.
- **Likelihood‑guided**: sample high‑likelihood trajectories under constraints.

---

## 3) Pseudocode
**Diffusion rollout (receding horizon)**
```
ctx ← encoder(obs_t, goal, hist)
a_K ~ N(0, I)
for k = K..1:
    eps_hat = eps_theta(a_k, k, ctx)
    a_{k-1} = denoise(a_k, eps_hat, β_k)
execute first h steps of a_0..a_{H-1}
shift window and repeat
```

**Trajectory transformer planning**
```
encode τ as tokens with returns
for t:
    sample a_t ~ p_θ(· | tokens_≤t)
    step env, append tokens
```

---

## 4) Evaluation & ablations
- Success rate, time‑to‑goal, collisions; long‑horizon tasks with distractors.
- Compare to BC/SAC using identical encoders.
- Ablate denoise steps $K$, schedule $\{\beta_k\}$, horizon $H$, and guidance strength.

---

## 5) Pitfalls
- Too few denoise steps → underfit; too many → slow inference.
- Mismatch between action parameterization and dynamics (e.g., positions vs. velocities).
- Visual overfitting without strong augmentation/DR.

---

## 6) Interview talking points
- “DP uses **score‑matching** in action space to represent **multimodal** policies; TT reframes control as **sequence modeling** and leverages LM inference tools.”

