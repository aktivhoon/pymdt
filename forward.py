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
        
        # State transition pairs dictionary
        self.state_pairs = {
            # 5: 40, 6: 20, 7: 10, 8: 0
            0: {0: [1, 2], 1: [3, 4]},
            1: {0: [6, 7], 1: [7, 8]},
            2: {0: [7, 8], 1: [6, 8]},
            3: {0: [6, 5], 1: [5, 8]},
            4: {0: [6, 8], 1: [8, 5]}
        }
        
        self.reset()

    def action(self, state):
        probs = softmax(self.Q_fwd[state], self.beta)
        return np.random.choice(self.num_actions, p=probs)
    
    def get_Q_values(self, state):
        if state >= self.output_offset:
            reward_idx = self.reward_map.index(self.reward_map[state - self.output_offset])
            state = reward_idx + self.output_offset
        return self.Q_fwd[state]

    def update(self, state, reward, action, next_state, current_goal, pretrain=False):
        spe = 1 - self.T[state][action][next_state]

        # Update transition model
        self.T[state][action][next_state] += self.learning_rate * spe
        
        for j in range(self.num_states):
            if j != next_state:
                self.T[state][action][j] *= (1 - self.learning_rate)
        
        # Update rewards if necessary
        if next_state >= self.output_offset and reward != self.observe_reward[next_state]:
            self.observe_reward[self.reward_map.index(reward) + self.output_offset] = reward
        
        # Q-value update: compute the new Q-value as a weighted sum over possible transitions.
        # The weight is the probability of each transition.
        temp_sum = 0
        temp_sum += self.T[state][action][next_state] * (reward + self.discount_factor * np.max(self.Q_fwd[next_state]))
        for candidate in range(self.num_states):
            if candidate == next_state:
                continue
            if pretrain or current_goal == -1 or current_goal == candidate:
                reward = self.observe_reward[candidate]
            else:
                reward = 0
                
            adding = self.T[state][action][candidate] * (reward + self.discount_factor * np.max(self.Q_fwd[candidate]))
            temp_sum += adding
        self.Q_fwd[state][action] = temp_sum

        return spe

    def reset(self):
        """Reset transition model and Q-values"""
        self.observe_reward = np.zeros(self.num_states)
        
        # Initialize transition probabilities uniformly for each state-action pair
        for state in range(self.num_states):
            self.T[state] = {}
            for action in range(self.num_actions):
                # T: state x action x next_state
                self.T[state][action] = {next_state: 1 / self.num_states for next_state in range(self.num_states)}

        self._Q_fitting()

    def _Q_fitting(self):
        """Update Q-values based on transition model"""
        self.Q_fwd = defaultdict(lambda: np.zeros(self.num_actions))
        
        for state in (range(self.num_states)):
            for action in range(self.num_actions):
                if state in self.state_pairs and action in self.state_pairs[state]:
                    self.Q_fwd[state][action] = 0

        self.policy_fn = lambda s: softmax(self.Q_fwd[s], self.beta)

    def backward_update(self, current_goal):
        """Update the transition model for a new goal context"""                                
        # Recompute Q-values
        for state in [1, 2, 3, 4, 0]:
            for action in range(self.num_actions):
                tmp = 0
                for next_state in range(self.num_states):
                    if current_goal == next_state or current_goal == -1:
                        reward = self.observe_reward[next_state]
                    else:
                        reward = 0
                    tmp += self.T[state][action][next_state] * (reward + self.discount_factor * np.max(self.Q_fwd[next_state]))
                self.Q_fwd[state][action] = tmp
        self.policy_fn = lambda s: softmax(self.Q_fwd[s], self.beta)