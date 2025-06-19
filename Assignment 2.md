# ECE 457C Assignment 2 Report

Authors: Jerome Philip, Jonathan Huo

# Descriptions of each RL algorithm with math and pseudocode

## 1. Monte Carlo (MC)

### Description

Monte Carlo methods estimate the action-value function $Q(s, a)$ by averaging the returns following each first visit to $(s, a)$ in an episode. It does not bootstrap — updates are made after complete episodes using sampled returns. This makes it suitable for episodic tasks where the episode eventually terminates.

### Mathematical Formulation

Let the return from time $t$ be:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+1+k}
$$

Then the update rule is:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( G_t - Q(s_t, a_t) \right)
$$

### Pseudocode

```
Initialize Q(s, a) arbitrarily
Loop over episodes:
    Generate an episode: s₀, a₀, r₁, s₁, a₁, ..., s_T
    For each (s, a) pair in the episode:
        G = return following the first occurrence of (s, a)
        Q(s, a) += α * (G - Q(s, a))
```

---

## 2. SARSA (On-Policy TD Control)

### Description

SARSA (State–Action–Reward–State–Action) is an on-policy algorithm that updates $Q(s, a)$ based on the action actually taken by the policy in the next state. This allows it to incorporate exploration into the learning updates.

### Mathematical Formulation

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

### Pseudocode

```
Initialize Q(s, a) arbitrarily
Loop over episodes:
    Initialize s
    Choose a from s using ε-greedy
    Repeat until s is terminal:
        Take action a, observe r and s'
        Choose a' from s' using ε-greedy
        Q(s, a) += α * (r + γ * Q(s', a') - Q(s, a))
        s ← s'; a ← a'
```

---

## 3. Q-Learning (Off-Policy TD Control)

### Description

Q-Learning is an off-policy TD control algorithm. It updates $Q(s, a)$ toward the maximum estimated value of the next state, regardless of the policy used to select the next action. This allows learning the optimal policy independently of the agent’s behaviour.

### Mathematical Formulation

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

### Pseudocode

```
Initialize Q(s, a) arbitrarily
Loop over episodes:
    Initialize s
    Repeat until s is terminal:
        Choose a from s using ε-greedy
        Take action a, observe r and s'
        Q(s, a) += α * (r + γ * max_a' Q(s', a') - Q(s, a))
        s ← s'
```

---

## 4. Expected SARSA

### Description

Expected SARSA is a hybrid between SARSA and Q-Learning. Instead of using the sampled action from the next state, it uses the expected value under the current policy. This results in lower variance and more stable learning.

### Mathematical Formulation

Let $\pi(a' \mid s')$ be the probability of taking action $a'$ in state $s'$ under the current policy. Then:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a' \mid s_{t+1}) Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

### Pseudocode

```
Initialize Q(s, a) arbitrarily
Loop over episodes:
    Initialize s
    Choose a from s using ε-greedy
    Repeat until s is terminal:
        Take action a, observe r and s'
        Compute expected value: E[Q(s', a')] under ε-greedy
        Q(s, a) += α * (r + γ * E[Q(s', a')] - Q(s, a))
        Choose a' from s' using ε-greedy
        s ← s'; a ← a'
```

---

## 5. TD($\lambda$) with Eligibility Traces

### Description

TD($\lambda$) combines TD learning with a memory of past state-action pairs using eligibility traces. This allows updates to be distributed to recently visited states, creating a bridge between Monte Carlo and TD(0) methods.

### Mathematical Formulation

-   Initialize all eligibility traces: $E(s,a) = 0$
-   For each step in the episode:

    -   TD error:

        $$
        \delta_t = R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        $$

    -   Update eligibility trace:

        $$
        E(s_t, a_t) \leftarrow E(s_t, a_t) + 1
        $$

    -   For all $(s, a)$:

        $$
        Q(s, a) \leftarrow Q(s, a) + \alpha \delta_t E(s, a)
        $$

        $$
        E(s, a) \leftarrow \gamma \lambda E(s, a)
        $$

### Pseudocode

```
Initialize Q(s, a), E(s, a) = 0
Loop over episodes:
    Initialize s
    Choose a using ε-greedy
    For each step of the episode:
        Take action a, observe r and s'
        Choose a' from s' using ε-greedy
        δ ← r + γ * Q(s', a') - Q(s, a)
        E(s, a) += 1
        For all (s, a):
            Q(s, a) += α * δ * E(s, a)
            E(s, a) *= γ * λ
        s ← s'; a ← a'
```

# Analysis

Below there are multiple plots comparing various metrics of the different algorithms.

As required from the assignment description, I have included the following plots:

1. Total reward per episode over time (**Mandatory**)
2. Episode length per episode over time (**Mandatory**)
3. Path length per episode over time (Optional)
4. Pit hits and wall hits count per episode over time (Optional)
5. Reward variance over time (Optional)

The optional ones are included to give better insight into the algorithms and their performance.

## 1 - Total reward per episode over time

For each of the 3 tasks I have built a plot comparing the summed reward per episode for each algorithm.

<TODO: Add plots>

## 2 - Episode length per episode over time

For each of the 3 tasks I have built a plot comparing the episode length per episode for each algorithm.

<TODO: Add plots>

## 3 - Path length per episode over time

For each of the 3 tasks I have built a plot comparing the path length per episode for each algorithm.

<TODO: Add plots>

## 4 - Pit hits and wall hits count per episode over time

For each of the 3 tasks I have built a plot comparing the pit hits and wall hits count per episode for each algorithm.

<TODO: Add plots>

## 5 - Reward variance over time

For each of the 3 tasks I have built a plot comparing the reward variance over time for each algorithm.

<TODO: Add plots>

# Qualitative comparisons between algorithms

<TODO: finish this section>
