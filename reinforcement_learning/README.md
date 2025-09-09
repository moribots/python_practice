# Reinforcement Learning Study Guide

Reinforcement Learning (RL) is crucial for robotics, enabling agents to learn through interaction with environments. Robotics interviews often test RL fundamentals, algorithmic understanding, and practical implementation skills.

## Key Concepts

### Q-Learning
- Value-based RL algorithm
- Q-table: state-action values
- Update: Q(s,a) = Q(s,a) + α[R + γ max(Q(s',a')) - Q(s,a)]

### Policy Evaluation
- Compute value function for a policy
- Iterative method
- Bellman equation: V(s) = Σ P(s'|s,a)[R + γ V(s')]

### ε-Greedy
- Exploration-exploitation balance
- With probability ε: random action
- With probability 1-ε: greedy action

### Monte Carlo
- Learn from complete episodes
- Value = average returns
- Works for non-Markov environments

## Interview-Ready Concepts

### Temporal Difference Learning
- TD(0): One-step bootstrapping
- SARSA: On-policy TD control
- Q-learning: Off-policy TD control
- More sample-efficient than Monte Carlo

### Function Approximation
- Use neural networks for large state spaces
- Deep Q-Networks (DQN)
- Policy gradients
- Actor-critic methods

### Exploration Strategies
- ε-greedy decay
- Boltzmann exploration
- Upper confidence bounds (UCB)
- Thompson sampling

## Worked Examples

### Problem 1: Q-Learning Update
```python
import numpy as np

def q_learning_update(Q, state, action, reward, next_state, alpha, gamma):
    current_q = Q[state, action]
    max_next_q = np.max(Q[next_state])
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    Q[state, action] = new_q
    return Q

# Test
Q = np.zeros((5, 4))  # 5 states, 4 actions
Q = q_learning_update(Q, 0, 0, 1, 1, 0.1, 0.9)
print(Q[0, 0])  # 0.1
```

### Problem 2: Policy Evaluation
```python
def policy_evaluation(policy, transitions, rewards, gamma, theta=1e-6):
    V = {s: 0 for s in policy.keys()}
    
    while True:
        delta = 0
        for s in policy.keys():
            v = V[s]
            action = policy[s]
            V[s] = sum(prob * (reward + gamma * V[next_s]) 
                       for prob, next_s, reward in transitions.get((s, action), []))
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V

# Simplified example
policy = {0: 0, 1: 1}
transitions = {(0, 0): [(1.0, 1, 0)], (1, 1): [(1.0, 1, 1)]}
rewards = {(0, 0, 1): 0, (1, 1, 1): 1}
V = policy_evaluation(policy, transitions, rewards, 0.9)
print(V)  # {0: 0.9, 1: 1.0}
```

### Problem 3: ε-Greedy Action Selection
```python
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

# Test
Q = np.array([[1, 0, 0], [0, 2, 0]])
action = epsilon_greedy(Q, 0, 0.1)
print(action)  # Likely 0 (greedy)
```

### Problem 4: Monte Carlo Value Estimation
```python
def monte_carlo_value(episodes, gamma):
    returns = {}
    counts = {}
    V = {}
    
    for episode in episodes:
        G = 0
        visited = set()
        
        # Reverse episode for first-visit MC
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = gamma * G + reward
            
            if state not in visited:
                visited.add(state)
                if state not in returns:
                    returns[state] = []
                    counts[state] = 0
                returns[state].append(G)
                counts[state] += 1
                V[state] = np.mean(returns[state])
    
    return V

# Example episode: [(state, reward), ...]
episodes = [[(0, 0), (1, 1), (1, 1)], [(0, 0), (1, 0), (1, 1)]]
V = monte_carlo_value(episodes, 0.9)
print(V)  # {0: 0.81, 1: 1.0}
```

## Advanced Interview Topics

### SARSA Implementation
```python
def sarsa_update(Q, state, action, reward, next_state, next_action, alpha, gamma):
    current_q = Q[state, action]
    next_q = Q[next_state, next_action]
    new_q = current_q + alpha * (reward + gamma * next_q - current_q)
    Q[state, action] = new_q
    return Q

# SARSA episode
def sarsa_episode(env, Q, epsilon, alpha, gamma):
    state = env.reset()
    action = epsilon_greedy(Q, state, epsilon)
    total_reward = 0
    
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = epsilon_greedy(Q, next_state, epsilon)
        
        Q = sarsa_update(Q, state, action, reward, next_state, next_action, alpha, gamma)
        
        state, action = next_state, next_action
        total_reward += reward
        
        if done:
            break
    
    return Q, total_reward
```

### Deep Q-Learning Basics
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### Policy Gradient Methods
```python
def reinforce_update(policy_net, optimizer, rewards, log_probs, gamma):
    # Compute discounted rewards
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    
    # Normalize rewards
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    # Compute policy loss
    policy_loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * reward)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
```

### Multi-Armed Bandit Problems
```python
# Upper Confidence Bound (UCB)
def ucb_action(counts, values, t):
    if 0 in counts:
        return counts.index(0)  # Try untried action
    
    ucb_values = []
    for i in range(len(counts)):
        bonus = np.sqrt(2 * np.log(t) / counts[i])
        ucb_values.append(values[i] + bonus)
    
    return np.argmax(ucb_values)

# Thompson Sampling
def thompson_sampling(alpha, beta):
    samples = [np.random.beta(a, b) for a, b in zip(alpha, beta)]
    return np.argmax(samples)
```

## Practice Tips
- Understand Markov Decision Processes (MDP)
- Balance exploration vs exploitation
- RL works best with good state representations
- Consider function approximation for large state spaces
- Use libraries like Gym for environment interaction
- Think about sample efficiency and computational constraints
- Robotics applications: manipulation, navigation, multi-agent systems
