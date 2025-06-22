# ECE 457C Assignment 2 Report

Authors: Jerome Philip, Jonathan Huo

# Descriptions of each RL algorithm

## 1. Monte Carlo (MC)

### Description

Monte Carlo methods update the action-value function $Q(s, a)$ based on complete episodes. For each state-action pair $(s, a)$, the algorithm waits until the episode finishes, computes the total return $G$ following the first time $(s, a)$ was visited, and uses this return to update $Q(s, a)$. No updates are made during the episode — only after it ends.

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

SARSA updates $Q(s, a)$ using the next action actually taken by the agent. After taking action $a$ in state $s$, receiving reward $r$, and transitioning to state $s'$, the algorithm selects the next action $a'$ from $s'$ and uses $Q(s', a')$ in the update. This process is repeated step-by-step throughout the episode.

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

SARSA updates $Q(s, a)$ using the next action actually taken by the agent. After taking action $a$ in state $s$, receiving reward $r$, and transitioning to state $s'$, the algorithm selects the next action $a'$ from $s'$ and uses $Q(s', a')$ in the update. This process is repeated step-by-step throughout the episode.


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

Expected SARSA updates $Q(s, a)$ using the expected value over all possible next actions under the current policy. After observing state $s'$ and the reward $r$, the algorithm computes the weighted average of $Q(s', a')$ for all $a'$ using the policy's action probabilities, and uses this expected value in the update.


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

TD($\lambda$) maintains an eligibility trace for each state-action pair, which tracks how recently and frequently it has been visited. At each step, the TD error is computed and used to update all $Q(s, a)$ values in proportion to their eligibility trace. The traces decay over time according to the discount factor $\gamma$ and trace decay rate $\lambda$.

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

Q-Learning showed strong performance across most tasks. It learned optimal policies quickly, with sharp improvements in both total rewards and episode lengths. The agent also reduced pit and wall collisions rapidly, suggesting it learned to avoid hazards efficiently. However, the early stages showed some instability, likely due to its off-policy nature, where it updates using the best possible next action rather than what was actually taken. This sometimes led it to overshoot safer paths while exploring. Overall, Q-Learning was aggressive and efficient once it stabilized, making it a strong performer when fast convergence is the goal.

## SARSA (On-Policy TD)

Compared to Q-Learning, SARSA had a more stable learning curve. Improvements in total reward and episode length happened more gradually, but consistently. Since it updates using the action actually taken, its behavior matched the agent’s exploration strategy, leading to fewer early mistakes like pit or wall hits. This more cautious learning came at the cost of slower convergence. In some cases, it appeared to settle for safer, possibly suboptimal paths rather than aggressively exploring to find the best one. SARSA is a solid choice when you want a method that’s less prone to instability and aligns better with what the agent actually does.

## Expected SARSA

Expected SARSA stood out for its consistency. It had very low variance in both reward and episode length over time, and its hazard metrics stayed stable and low throughout training. By averaging over all possible next actions, it smoothed out the learning updates and avoided large spikes. Although it didn’t always reach the peak performance of Q-Learning, it was more predictable and steady. This method is especially useful when you want reliable learning with fewer surprises, even if it means sacrificing a bit of speed.

## TD($\lambda$)

TD($\lambda$) performed the best in terms of how quickly it improved both rewards and episode lengths. The use of eligibility traces let it apply updates to many states at once, which helped it generalize faster from recent experience. This led to steep gains early in training. That said, it did show some moderate variance at the start, and tuning the $\lambda$ parameter added an extra layer of complexity. Still, for tasks where past decisions influence current ones — like long paths or sequences — TD($\lambda$) was a strong fit and learned quickly.


## Monte Carlo

Monte Carlo learning had a very different behavior compared to the TD methods. Since it only updates after full episodes, its learning curve was slower and more variable. We saw higher variance in rewards and episode lengths, especially early on. It also struggled more in tasks with longer episodes or higher noise. But because it updates using the actual return over an entire episode, its estimates were more grounded in full experience. This makes it more suitable when the episodes are well-bounded and you want exact return values, even if the learning is a bit less efficient.


