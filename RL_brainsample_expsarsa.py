import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        self.lr = kwargs.get('learning_rate', 0.01)
        self.gamma = kwargs.get('reward_decay', 0.9)
        self.epsilon = kwargs.get('e_greedy', 0.1)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = "Expected SARSA"
        self.maxq = 0
        self.delta = 0

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0.0] * len(self.actions)

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

        q_predict = self.q_table.loc[s, a]

        q_next = self.q_table.loc[s_, :]

        max_action = q_next.idxmax()
        expected_q = 0

        for action in self.actions:
            prob = self.epsilon / len(self.actions)
            if action == max_action:
                prob += 1 - self.epsilon
            expected_q += prob * q_next[action]

        q_target = r + self.gamma * expected_q

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

        self.delta = max(self.delta, abs(q_target - q_predict))
        self.maxq = max(self.maxq, abs(self.q_table.loc[s, a]))

        return s_, None
