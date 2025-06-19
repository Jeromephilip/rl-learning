import numpy as np
import pandas as pd

class rlalgorithm:
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        self.lr = kwargs.get('learning_rate', 0.01)
        self.gamma = kwargs.get('reward_decay', 0.9)
        self.epsilon = kwargs.get('e_greedy', 0.1)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = 'SARSA'
        self.maxq = 0
        self.delta = 0

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0] * len(self.actions)

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_actions = self.q_table.loc[observation]
            action = state_actions.idxmax()
        return action

    def learn(self, s, a, r, s_, **kwargs):
        self.check_state_exist(s)
        self.check_state_exist(s_)

        a_ = self.choose_action(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, a_]
        self.delta = abs(q_predict - q_target)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

        self.maxq = max(self.maxq, self.q_table.loc[s, a])
        return s_, a_

    def count_state(self, s):
        return len(self.q_table), s, self.q_table.index.tolist().count(s)
