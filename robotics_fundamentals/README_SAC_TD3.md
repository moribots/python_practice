# Soft Actor–Critic (SAC) and Twin‑Delayed DDPG (TD3)

## TL;DR
- **SAC** maximizes reward **plus** policy entropy via a learned temperature $\alpha$; stochastic, robust, and sample‑efficient.
- **TD3** fixes DDPG’s overestimation with **twin critics**, **target policy smoothing**, and **delayed actor updates**; deterministic.

---

## 1) Problem setup (continuous control)
We consider a Markov Decision Process (MDP): states $s_t\in\mathbb{R}^{d_s}$, actions $a_t\in\mathbb{R}^{d_a}$, reward $r(s_t,a_t)\in\mathbb{R}$, transitions $p(s_{t+1}\mid s_t,a_t)$, discount $\gamma\in(0,1)$.

**Standard RL objective**
$$
J(\pi) \,=\, \mathbb{E}_{\pi}\Big[\sum_{t=0}^{\infty} \gamma^t \, r(s_t,a_t)\Big].
$$

### Maximum‑entropy RL (SAC)
Augment return with the entropy $\mathcal{H}(\pi(\cdot\mid s))$:
$$
J_{\text{soft}}(\pi) \,=\, \mathbb{E}_{\pi}\Big[\sum_{t=0}^{\infty} \gamma^t\big( r(s_t,a_t) + \alpha \, \mathcal{H}(\pi(\cdot\mid s_t))\big)\Big].
$$
Where:
- $\alpha>0$: temperature that trades reward vs. exploration.

---

## 2) SAC from first principles
### Policy parameterization
- Pre‑squash Gaussian with mean $\mu_\theta(s)$ and diagonal std $\sigma_\theta(s)$, then $\tanh$ squashing.
- Reparameterize: draw $\epsilon\sim\mathcal{N}(0,I)$, set $z = \mu_\theta(s) + \sigma_\theta(s)\odot\epsilon$, and $a=\tanh(z)$.

**Log‑prob with squashing correction**
$$
\log \pi_\theta(a\mid s)
\,=\, \log\mathcal{N}\big(z;\,\mu_\theta,\sigma_\theta\big)
\;-
\sum_i \log\big(1-\tanh^2(z_i)+\varepsilon\big).
$$

### Soft Q and actor losses
Target using twin target critics $Q_{\bar\phi_1}, Q_{\bar\phi_2}$:
$$
\begin{aligned}
& a'\sim\pi_\theta(\cdot\mid s'),\ \ \ \n y \,=\, r + \gamma (1-d)\,\Big( \min_i Q_{\bar\phi_i}(s',a') - \alpha \log\pi_\theta(a'\mid s') \Big).\\
&\mathcal{L}_{Q_i} \,=\, \mathbb{E}\big[\big(Q_{\phi_i}(s,a) - y\big)^2\big],\quad i\in\{1,2\}.\\[2mm]
&\mathcal{L}_\pi \,=\, \mathbb{E}_{s\sim\mathcal{D}}\Big[\alpha \log\pi_\theta(a\mid s) - \min_i Q_{\phi_i}(s,a)\Big],\quad a\sim\pi_\theta(\cdot\mid s).
\end{aligned}
$$

### Automatic temperature tuning
Choose a target entropy $\mathcal{H}_\text{targ}\approx -d_a$ and learn $\alpha$ by
$$
\mathcal{L}_\alpha \,=\, -\,\mathbb{E}_{a\sim\pi}\big[\alpha\,(\log\pi_\theta(a\mid s)+\mathcal{H}_\text{targ})\big].
$$

### Target network update (Polyak averaging)
$$
\bar\phi \leftarrow \tau\,\phi + (1-\tau)\,\bar\phi,\qquad \tau\ll1.
$$

---

## 3) TD3 essentials
- **Twin critics:** use $\min(Q_1,Q_2)$ for targets to reduce overestimation.
- **Target policy smoothing:** $a' = \pi_{\bar\theta}(s') + \operatorname{clip}(\epsilon,[-c,c])$, $\epsilon\sim\mathcal{N}(0,\sigma^2 I)$.
- **Delayed actor:** update actor and targets every $d$ critic steps.

**TD3 target**
$$
 y \,=\, r + \gamma (1-d) \min\big(Q_{\bar\phi_1}(s',a'),\,Q_{\bar\phi_2}(s',a')\big).
$$

---

## 4) Pseudocode (per gradient step)
**SAC**
```
sample (s,a,r,s',d) from replay
# critics
a2, logp2 = policy.sample(s')
y = r + γ(1-d) * ( min(Q1_t(s',a2), Q2_t(s',a2)) - α*logp2 )
step Q1,Q2 to fit y
# actor & α
ap, logpp = policy.sample(s)
loss_pi = mean( α*logpp - min(Q1(s,ap), Q2(s,ap)) )
step actor
loss_alpha = mean( -(log_alpha) * (logpp + H_target) )
step α
soft_update(targets)
```

**TD3**
```
a_targ = π_targ(s') + clip(N(0,σ),[-c,c])
y = r + γ(1-d) * min(Q1_t(s',a_targ), Q2_t(s',a_targ))
step Q1,Q2
if step % delay == 0:
    step actor on -Q1(s, π(s))
    soft_update targets
```

---

## 5) Evaluation & ablations
- Learning curves (mean ± 95% CI over seeds), policy entropy, $\alpha$ trajectory, Q‑targets scale.
- Ablate: twin‑critic to single, turn off target smoothing, fixed vs. learned $\alpha$.

## 6) Common pitfalls
- Missing $\tanh$ log‑det correction; unclamped log‑std.
- Not stopping gradients through target computations.
- Warm‑up too short → critic divergence.

## 7) Interview talking points
- “SAC’s **soft** objective yields exploration calibrated by a **learned** temperature to hit a target entropy.”
- “TD3 stabilizes bootstrapping with **min of twins**, **smoothing noise**, and **delayed** actor updates.”

