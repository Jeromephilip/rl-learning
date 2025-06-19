import numpy as np
import pandas as pd
from collections import defaultdict

class rlalgorithm:
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        self.lr = kwargs.get('learning_rate', 0.01)
        self.gamma = kwargs.get('reward_decay', 0.9)
        self.epsilon = kwargs.get('e_greedy', 0.1)

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.returns = defaultdict(list)
        self.episode = []
        self.display_name = "MC Control (On-policy)"
        self.maxq = 0
        self.delta = 0

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0.0] * len(self.actions)

    def choose_action(self, state, **kwargs):
        self.check_state_exist(state)

        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[state, :]
            return np.random.choice(state_action[state_action == state_action.max()].index)

    def learn(self, s, a, r, s_, **kwargs):
        done = kwargs.get("done", False)
        self.episode.append((s, a, r))

        if not done:
            return s_, None

        G = 0
        visited = set()

        for t in reversed(range(len(self.episode))):
            s_t, a_t, r_t = self.episode[t]
            G = self.gamma * G + r_t

            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                self.check_state_exist(s_t)

                self.returns[(s_t, a_t)].append(G)
                avg_return = np.mean(self.returns[(s_t, a_t)])
                old_q = self.q_table.loc[s_t, a_t]
                self.q_table.loc[s_t, a_t] = avg_return

                self.maxq = max(self.maxq, abs(avg_return))
                self.delta = max(self.delta, abs(old_q - avg_return))

        self.episode = []
        return s_, None

    def count_state(self, s):
        self.check_state_exist(s)
        return len(self.q_table), s, self.q_table.index.tolist().count(s)
