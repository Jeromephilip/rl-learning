import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        self.lr = kwargs.get('learning_rate', 0.01)
        self.gamma = kwargs.get('reward_decay', 0.9)
        self.epsilon = kwargs.get('e_greedy', 0.1)
        self.lambda_ = kwargs.get('lambda_', 0.9)

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.elig_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        self.display_name = "TD(Lambda)"
        self.maxq = 0
        self.delta = 0

    def check_state_exist(self, state):
        for table in [self.q_table, self.elig_table]:
            if state not in table.index:
                table.loc[state] = [0.0] * len(self.actions)

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[observation, :]
            return np.random.choice(state_action[state_action == state_action.max()].index)

    def learn(self, s, a, r, s_, **kwargs):
        self.check_state_exist(s)
        self.check_state_exist(s_)

        a_ = self.choose_action(s_)
        
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, a_]
        delta = q_target - q_predict

        self.elig_table.loc[s, a] += 1

        for state in self.q_table.index:
            for action in self.actions:
                self.q_table.loc[state, action] += self.lr * delta * self.elig_table.loc[state, action]
                self.elig_table.loc[state, action] *= self.gamma * self.lambda_

        self.maxq = max(self.maxq, abs(self.q_table.loc[s, a]))
        self.delta = max(self.delta, abs(delta))

        return s_, a_

    def count_state(self, s):
        self.check_state_exist(s)
        return len(self.q_table), s, self.q_table.index.tolist().count(s)
