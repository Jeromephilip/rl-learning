# ECE 457C Assignment 2 Report

Authors: Jerome Philip, Jonathan Huo

# Descriptions of each RL algorithm with math and pseudocode

## 1. Monte Carlo (MC)

### Description

Monte Carlo methods estimate the[text](<Assignment 2.md>) action-value function $Q(s, a)$ by averaging the returns following each first visit to $(s, a)$ in an episode. It does not bootstrap — updates are made after complete episodes using sampled returns. This makes it suitable for episodic tasks where the episode eventually terminates.

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

# Quantitative Analysis (Graphs)

Below there are multiple plots comparing various metrics of the different algorithms.

As required from the assignment description, I have included the following plots:

1. Total reward per episode over time and Episode length per episode over time (**Mandatory**)
2. Pit hits and wall hits count per episode over time (Optional)

The optional ones are included to give better insight into the algorithms and their performance.

## 1 - Total reward per episode over time

### QLearning

![qlearning_task_1_rewards_and_lengths](report_attachments/qlearning/T1-d20250620_t171702-LR0.01_gamma0.9_eps0.1.png)

![qlearning_task_2_rewards_and_lengths](report_attachments/qlearning/T2-d20250620_t171701-LR0.01_gamma0.9_eps0.1.png)

![qlearning_task_3_rewards_and_lengths](report_attachments/qlearning/T3-d20250620_t171717-LR0.01_gamma0.9_eps0.1.png)

### SARSA

![sarsa_task_1_rewards_and_lengths](report_attachments/sarsa/T1-d20250620_t172738-LR0.01_gamma0.9_eps0.1.png)

![sarsa_task_2_rewards_and_lengths](report_attachments/sarsa/T2-d20250620_t172609-LR0.01_gamma0.9_eps0.1.png)

![sarsa_task_3_rewards_and_lengths](report_attachments/sarsa/T3-d20250620_t172624-LR0.01_gamma0.9_eps0.1.png)

### Expected SARSA

![e_sarsa_task_1_rewards_and_lengths](report_attachments/expected_sarsa/T1-d20250620_t173519-LR0.01_gamma0.9_eps0.1.png)

![e_sarsa_task_2_rewards_and_lengths](report_attachments/expected_sarsa/T2-d20250620_t173511-LR0.01_gamma0.9_eps0.1.png)

![e_sarsa_task_3_rewards_and_lengths](report_attachments/expected_sarsa/T3-d20250620_t173635-LR0.01_gamma0.9_eps0.1.png)

### TD($\lambda$)

![td_task_1_rewards_and_lengths](report_attachments/td/T1-d20250620_t173859-LR0.01_gamma0.9_eps0.1.png)

![td_task_2_rewards_and_lengths](report_attachments/td/T2-d20250620_t173739-LR0.01_gamma0.9_eps0.1.png)

![td_task_3_rewards_and_lengths](report_attachments/td/T3-d20250620_t174030-LR0.01_gamma0.9_eps0.1.png)

### Monte Carlo

![mc_task_1_rewards_and_lengths](report_attachments/mc/T1-d20250620_t180145-LR0.01_gamma0.9_eps0.1.png)

![mc_task_2_rewards_and_lengths](report_attachments/mc/T2-d20250620_t174155-LR0.01_gamma0.9_eps0.1.png)

![mc_task_3_rewards_and_lengths](report_attachments/mc/T3-d20250620_t174218-LR0.01_gamma0.9_eps0.1.png)

## 2 - Pit hits and wall hits count per episode over time

### QLearning

![qlearning_task_1_hits](report_attachments/qlearning/T1-d20250620_t171702-LR0.01_gamma0.9_eps0.1_hits.png)

![qlearning_task_2_hits](report_attachments/qlearning/T2-d20250620_t171701-LR0.01_gamma0.9_eps0.1_hits.png)

![qlearning_task_3_hits](report_attachments/qlearning/T3-d20250620_t171717-LR0.01_gamma0.9_eps0.1_hits.png)

### SARSA

![sarsa_task_1_hits](report_attachments/sarsa/T1-d20250620_t172738-LR0.01_gamma0.9_eps0.1_hits.png)

![sarsa_task_2_hits](report_attachments/sarsa/T2-d20250620_t172609-LR0.01_gamma0.9_eps0.1_hits.png)

![sarsa_task_3_hits](report_attachments/sarsa/T3-d20250620_t172624-LR0.01_gamma0.9_eps0.1_hits.png)

### Expected SARSA

![esarsa_task_1_hits](report_attachments/expected_sarsa/T1-d20250620_t173519-LR0.01_gamma0.9_eps0.1_hits.png)

![esarsa_task_2_hits](report_attachments/expected_sarsa/T2-d20250620_t173511-LR0.01_gamma0.9_eps0.1_hits.png)

![esarsa_task_3_hits](report_attachments/expected_sarsa/T3-d20250620_t173635-LR0.01_gamma0.9_eps0.1_hits.png)

### TD($\lambda$)

![td_task_1_hits](report_attachments/td/T1-d20250620_t173859-LR0.01_gamma0.9_eps0.1_hits.png)

![td_task_2_hits](report_attachments/td/T2-d20250620_t173739-LR0.01_gamma0.9_eps0.1_hits.png)

![td_task_3_hits](report_attachments/td/T3-d20250620_t174030-LR0.01_gamma0.9_eps0.1_hits.png)

### Monte Carlo

![mc_task_1_hits](report_attachments/mc/T1-d20250620_t180145-LR0.01_gamma0.9_eps0.1_hits.png)

![mc_task_2_hits](report_attachments/mc/T2-d20250620_t174155-LR0.01_gamma0.9_eps0.1_hits.png)

![mc_task_3_hits](report_attachments/mc/T3-d20250620_t174218-LR0.01_gamma0.9_eps0.1_hits.png)

# Qualitative Analysis

I've put some comments with qualitative assessment of the performance of each algorithm and suggestions for use cases of each algorithm below.

# Qualitative Analysis

Based on the visual results from the total reward, episode length, and pit/wall hit graphs, we can assess the strengths and weaknesses of each reinforcement learning algorithm as follows:

## Q-Learning

**Strengths**:
- Rapid convergence in total rewards and reduced episode lengths across tasks.
- Learns optimal policy aggressively due to off-policy updates.
- Fast decrease in pit and wall hits shows efficient hazard avoidance.

**Weaknesses**:
- Initial instability due to aggressive exploration.
- Can overshoot safe paths early on.

**Best Use Case**: Suitable for environments where the agent must quickly learn an optimal policy, even at the cost of initial instability.

---

## SARSA (On-Policy TD)

**Strengths**:
- More stable and conservative learning compared to Q-Learning.
- Smoother and steadier improvement in rewards and episode lengths.
- Fewer early pit/wall hits due to alignment with the agent’s behavior policy.

**Weaknesses**:
- Slower convergence to optimal policy.
- May settle for suboptimal but safer policies.

**Best Use Case**: When safety is a concern and it’s important to learn based on actual actions taken (e.g., risky navigation).

---

## Expected SARSA

**Strengths**:
- Lowest variance in rewards and episode lengths across episodes.
- Updates are smoother due to averaging over expected actions.
- Consistently low and stable pit/wall hit metrics.

**Weaknesses**:
- Slightly slower than Q-Learning to reach peak performance.
- Requires knowledge of the action distribution under the policy.

**Best Use Case**: When stable, low-variance learning is desired, such as in sensitive environments or production systems.

---

## TD($\lambda$)

**Strengths**:
- Fastest convergence in most tasks due to eligibility traces.
- Leverages past experiences to speed up updates over multiple states.
- Episode lengths and rewards improved sharply and early.

**Weaknesses**:
- Slightly more complex to tune (due to $\lambda$ parameter).
- Moderate variance early on.

**Best Use Case**: When a balance between Monte Carlo and TD learning is needed, especially in tasks with temporal dependencies.

---

## Monte Carlo

**Strengths**:
- Does not require a model of the environment.
- Accurate return estimation over full episodes.

**Weaknesses**:
- High variance in both rewards and episode length.
- Slower convergence compared to TD-based methods.
- Less effective in environments with long episodes or high noise.

**Best Use Case**: Environments with well-defined episodes and when exact returns are important.


