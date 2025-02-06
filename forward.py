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

    def backward_update(self, current_goal):
        """
        Update the transition model and re-compute Q-values to adapt to a new goal context.
        
        The note specifies that:
          - When current_goal == -1 (flexible), then all outcome states (with indices >= output_offset)
            use their default rewards from self.reward_map.
          - When current_goal is one of [6, 7, 8], then only the outcome corresponding to that goal should
            yield its associated reward while all other outcome transitions are set to 0.
          
        (NOTE: Here we assume that in nine-state mode, outcome states are 5,6,7,8.
         According to the note, for example, if current_goal == 6 then outcome state 6 is the 40 reward state,
         if current_goal == 7 then outcome state 7 is the 20 reward state, and if current_goal == 8 then outcome state 8
         is the 10 reward state.)
        """
        for state in range(self.num_states):
            for action in range(self.num_actions):
                new_trans = []
                for next_state in range(self.num_states):
                    prob, _ = self.T[state][action][next_state]
                    # Only modify rewards for outcome states.
                    if next_state >= self.output_offset:
                        if current_goal == -1:
                            # Flexible: use the default reward from reward_map.
                            reward = self.reward_map[next_state - self.output_offset]
                        else:
                            # Specific goal: only the matching outcome state gets its designated reward.
                            if next_state == current_goal:
                                # Map outcome state to its proper reward.
                                reward = self.reward_map[next_state - self.output_offset]
                            else:
                                reward = 0
                    else:
                        reward = 0
                    new_trans.append((prob, reward))
                self.T[state][action] = new_trans
        # Recompute Q-values across the whole state space.
        self._Q_fitting()
