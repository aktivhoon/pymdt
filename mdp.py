# mdp.py
import numpy as np
import gym
from gym import spaces
from numpy.random import choice

class MDP(gym.Env):
    """Simplified MDP Environment"""
    
    HUMAN_AGENT_INDEX = 0
    STAGES = 2
    TRANSITION_PROBABILITY = [0.9, 0.1]
    NUM_ACTIONS = 2
    POSSIBLE_OUTPUTS = [0, 10, 20, 40]
    NINE_STATES_MODE = True

    def __init__(self, stages=STAGES, trans_prob=TRANSITION_PROBABILITY, 
                 num_actions=NUM_ACTIONS, outputs=POSSIBLE_OUTPUTS):
        # Environment setup
        self.stages = stages
        self.human_state = 0
        self.nine_states_mode = self.NINE_STATES_MODE
        
        # Reward setup
        self.reward_map = [0, 10, 20, 40]  # corresponding to states 5, 6, 7, 8
        self.reward_map_copy = self.reward_map.copy()
        
        # Action and state space setup
        self.action_space = [spaces.Discrete(num_actions)]
        self.trans_prob = trans_prob
        self.possible_actions = len(self.trans_prob) * num_actions
        self.outputs = outputs
        self.num_output_states = pow(self.possible_actions, self.stages)
        
        # Initialize output states
        self.output_states = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
        self.output_states_indx = [self.reward_map.index(x) for x in self.output_states]
        self.output_states_copy = self.output_states.copy()
        
        # Calculate state space
        self.output_states_offset = int((pow(self.possible_actions, self.stages) - 1) 
                                      / (self.possible_actions - 1))
        self.num_states = 9 if self.nine_states_mode else self.output_states_offset + self.num_output_states
        if self.nine_states_mode:
            self.output_states_offset = 5
            
        self.observation_space = [spaces.Discrete(self.num_states)]
        
        # Goal state tracking
        self.is_flexible = 1
        self.visited_goal_state = -1
        self.bwd_idf = -1

    def step(self, action):
        """Take one step in the environment based on the action

        Args:
            action: tuple (agent_index, action_choice)
        
        Returns:
            tuple: (new_state, reward, done, info)
        """
        if action[0] != self.HUMAN_AGENT_INDEX:
            raise ValueError("Only human agent actions are supported")
            
        state = self.human_state * self.possible_actions + \
                choice(range(action[1] * len(self.trans_prob) + 1,
                           (action[1] + 1) * len(self.trans_prob) + 1),
                      1, True, self.trans_prob)[0]
                
        reward = self._get_reward(state)
        
        if self.nine_states_mode and state > 4:
            state = self.output_states_indx[state-5] + 5
            
        self.human_state = state
        done = state >= self.output_states_offset
        
        if done:
            self.visited_goal_state = state
            
        return self.human_state, reward, done, {}

    def _get_reward(self, state):
        """Calculate reward for given state"""
        if state >= self.output_states_offset:
            return self.output_states[state - self.output_states_offset]
        return 0

    def reset(self):
        """Reset the environment
        
        Returns:
            int: Initial state
        """
        self.human_state = 0
        self.visited_goal_state = -1
        return self.human_state

    def shift_high_low_uncertainty(self):
        """Shift between high and medium uncertainty"""
        if self.trans_prob[0] == 0.5:
            self.trans_prob = [0.9, 0.1]
        elif self.trans_prob[0] == 0.9:
            self.trans_prob = [0.5, 0.5]
        else:
            raise ValueError