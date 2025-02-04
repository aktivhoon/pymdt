import numpy as np
from collections import defaultdict

def softmax(Q, beta):
    if isinstance(Q, dict):
        Q = np.array(list(Q.values()))
    elif isinstance(Q, list):
        Q = np.array(Q)
    Q_exp = np.exp(Q * beta)
    return Q_exp / np.sum(Q_exp, axis=0)

class FORWARD:
    def __init__(self, num_states, num_actions, output_offset, reward_map, 
                 beta=0.5, learning_rate=0.5, discount_factor=1.0):
        self.beta = beta
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.output_offset = output_offset
        self.reward_map = reward_map
        
        self.observe_reward = np.zeros(self.num_states)
        self.bwd_idf = -1
        
        # Initialize transition model and Q-values
        self.T = {}
        self.Q_fwd = defaultdict(lambda: np.zeros(self.num_actions))
        self.policy_fn = None
        self.reset()

    def action(self, state):
        probs = softmax(self.Q_fwd[state], self.beta)
        return np.random.choice(self.num_actions, p=probs)
    
    def get_Q_values(self, state):
        if state >= self.output_offset:
            reward_idx = self.reward_map.index(self.reward_map[state - self.output_offset])
            state = reward_idx + self.output_offset
        return self.Q_fwd[state]

    def optimize(self, state, reward, action, next_state, is_flexible=True):
        if state >= self.output_offset:
            reward_idx = self.reward_map.index(self.reward_map[state - self.output_offset])
            state = reward_idx + self.output_offset
        if next_state >= self.output_offset:
            reward_idx = self.reward_map.index(self.reward_map[next_state - self.output_offset])
            next_state = reward_idx + self.output_offset
            
        trans_prob = self.T[state][action]
        spe = 0
        
        for post_state in range(self.num_states):
            prob, r = trans_prob[post_state]
            if post_state == next_state:
                if next_state >= self.output_offset:
                    if not is_flexible:
                        if reward != self.observe_reward[next_state] and next_state != self.bwd_idf:
                            self.observe_reward[self.reward_map.index(reward) + self.output_offset] = reward
                            r = reward
                    else:
                        if reward != self.observe_reward[next_state]:
                            self.observe_reward[self.reward_map.index(reward) + self.output_offset] = reward
                            r = reward
                spe = 1 - prob
                trans_prob[post_state] = (prob + self.learning_rate * spe, r)
            else:
                trans_prob[post_state] = (prob * (1 - self.learning_rate), r)
        
        self.T[state][action] = trans_prob
        self._Q_fitting()
        return spe

    def reset(self):
        """Reset transition model and Q-values"""
        self.observe_reward = np.zeros(self.num_states)
        
        for state in range(self.num_states):
            self.T[state] = {action: [] for action in range(self.num_actions)}
            
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    if next_state == 0:
                        prob = 0
                    elif next_state < self.output_offset:
                        prob = 1.0 / (self.output_offset - 1)
                    else:
                        prob = 1.0 / (self.num_states - self.output_offset)
                        
                    reward = self.reward_map[next_state - self.output_offset] if next_state >= self.output_offset else 0
                    self.T[state][action].append((prob, reward))
                    
        self._Q_fitting()

    def _Q_fitting(self):
        """Update Q-values based on transition model"""
        self.Q_fwd = defaultdict(lambda: np.zeros(self.num_actions))
        for state in reversed(range(self.num_states)):
            for action in range(self.num_actions):
                for next_state in reversed(range(self.num_states)):
                    prob, reward = self.T[state][action][next_state]
                    if state >= self.output_offset:
                        reward = 0
                    best_action_value = np.max(self.Q_fwd[next_state])
                    self.Q_fwd[state][action] += prob * (reward + self.discount_factor * best_action_value)

        self.policy_fn = lambda s: softmax(self.Q_fwd[s], self.beta)

    def bwd_update(self, bwd_idf):
        self.bwd_idf = bwd_idf