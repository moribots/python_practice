# Behavior Cloning (BC) and DAgger (Dataset Aggregation)

## TL;DR
- **BC**: supervised learning of $\pi_\theta(a\mid s)$ from expert pairs $(s,a)$. Fails under **covariate shift** at test time.
- **DAgger**: iteratively collect states from the **learner**, label with **expert**, aggregate dataset, retrain → mitigates compounding error.

---

## 1) Problem setup
We have expert policy $\pi^*(a\mid s)$, and a dataset of demonstrations $\mathcal{D}=\{(s_i,a_i)\}$. For continuous actions, a deterministic BC objective is
$$
\min_{\theta} \ \mathbb{E}_{(s,a)\sim \mathcal{D}} \big[\,\|\pi_\theta(s)-a\|_2^2\,\big].
$$
Where:
- $\pi_\theta: \mathcal{S}\to\mathcal{A}$: policy network.
- $\mathcal{D}$: distribution induced by **expert** rollouts, not learner.

### Covariate shift & compounding error
Let per‑step error be $\epsilon$. Under BC, distribution mismatch yields total cost that can scale like $\mathcal{O}(T^2\epsilon)$ with horizon $T$.

---

## 2) DAgger from first principles
**Key idea**: train on the distribution of states **visited by the learner**.

**Algorithm**
1. Initialize $\mathcal{D}$ with $K$ expert rollouts.
2. Train $\pi_\theta$ on $\mathcal{D}$.
3. For $N$ iterations:
   - Roll out **learner** to collect states $S_{\text{learner}}$.
   - Query expert to label actions $A^* = \{a^*_s : s\in S_{\text{learner}}\}$.
   - Aggregate $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s,a^*_s)\}$.
   - Retrain $\pi_\theta$ on updated $\mathcal{D}$.

**Guarantee (intuition)**: if the supervised learner is no‑regret, DAgger’s error scales near‑linearly with $T$, eliminating BC’s compounding error.

---

## 3) Modeling choices
- **Stochastic BC**: model $\pi_\theta(a\mid s)$ as Gaussian; minimize negative log‑likelihood.
- **Multimodality**: use mixture density networks or diffusion policies (see separate README).
- **Regularization**: data augmentation (noise, random crops), weight decay, early stopping.

---

## 4) Pseudocode
```
D ← collect_expert(K episodes)
for it in 1..N:
    θ ← argmin_θ E_{(s,a)∈D} ℓ(π_θ(s), a)
    S_roll ← rollout(π_θ)
    A_star ← expert_labels(S_roll)
    D ← D ∪ {(S_roll, A_star)}
return π_θ
```

---

## 5) Evaluation protocol
- **Returns** and **success rate** vs. expert.
- **Action error** on (i) held‑out expert states and (ii) **learner‑visited** states.
- **Interventions**: number of expert corrections per rollout (human‑in‑the‑loop case).

---

## 6) Practical pitfalls
- Expensive expert labels → use probabilistic query schedules or confidence‑based querying.
- Dataset imbalance over time → reweight by iteration or use replay‑style sampling.
- Stochastic experts → learn distributions (NLL) rather than point estimates.

---

## 7) Interview talking points
- “BC fails because test‑time states are **policy‑dependent**. DAgger trains on the **learner‑induced** distribution, turning control into an online no‑regret problem.”

