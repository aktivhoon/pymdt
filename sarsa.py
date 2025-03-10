import numpy as np
from collections import defaultdict

def softmax(Q, beta):
    if isinstance(Q, dict):
        Q = np.array(list(Q.values()))
    elif isinstance(Q, list):
        Q = np.array(Q)
    Q_exp = np.exp(Q * beta)
    return Q_exp / np.sum(Q_exp, axis=0)

class SARSA:
    def __init__(self, num_actions, output_offset, reward_map,
                 beta=0.5, learning_rate=0.2, discount_factor=1.0):
        self.beta = beta
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.output_offset = output_offset
        self.reward_map = reward_map
        
        self.reset()

    def action(self, state):
        probs = softmax(self.Q_sarsa[state], self.beta)
        return np.random.choice(self.num_actions, p=probs)

    def get_Q_values(self, state):
        return self.Q_sarsa[state]

    def update(self, state, action_taken, reward, next_state, next_action=None):
        if next_action is None:
            next_action = self.action(next_state)
        rpe = self._get_rpe(reward, action_taken, next_action, state, next_state)
        self.Q_sarsa[state][action_taken] += self.learning_rate * rpe
        return rpe

    def _get_rpe(self, reward, action_taken, next_action, state, next_state):
        if next_action == -1:
            return reward - self.Q_sarsa[state][action_taken]
        else:
            return (reward + 
                    self.discount_factor * self.Q_sarsa[next_state][next_action] - 
                    self.Q_sarsa[state][action_taken])

    def reset(self):
        self.Q_sarsa = defaultdict(lambda: np.zeros(self.num_actions))
        for state in range(9):
            self.Q_sarsa[state] = np.zeros(self.num_actions)

    def copy(self, other):
        for state in range(9):
            for action in range(self.num_actions):
                self.Q_sarsa[state][action] = other.Q_sarsa[state][action]