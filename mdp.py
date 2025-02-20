# mdp.py
import numpy as np
import gym
from gym import spaces
from numpy.random import choice

class MDP(gym.Env):
    """Simplified MDP Environment"""
    
    HUMAN_AGENT_INDEX = 0
    STAGES = 2
    TRANSITION_PROBABILITY = [0.5, 0.5]
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
        self.reward_map = [40, 20, 10, 0]  # corresponding to states 5, 6, 7, 8
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

        # State transition mappings
        self.state_transitions = {
            0: {  # From state 0
                0: [1, 2],      # Action 0 leads to states 1 or 2
                1: [3, 4]       # Action 1 leads to states 3 or 4
            },
            1: {  # From state 1
                0: [7, 8],      # Action 0 leads to states 7 or 8
                1: [8, 5]       # Action 1 leads to states 8 or 5
            },
            2: {  # From state 2
                0: [8, 5],      # Action 0 leads to states 8 or 5
                1: [7, 5]       # Action 1 leads to states 7 or 5
            },
            3: {  # From state 3
                0: [7, 6],      # Action 0 leads to states 7 or 6
                1: [6, 5]       # Action 1 leads to states 6 or 5
            },
            4: {  # From state 4
                0: [7, 5],      # Action 0 leads to states 7 or 5
                1: [5, 6]       # Action 1 leads to states 5 or 6
            }
        }
        
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
            
        current_state = self.human_state
        action_choice = action[1]
        
        # Handle state transitions based on current state and action
        if current_state in self.state_transitions:
            possible_next_states = self.state_transitions[current_state][action_choice]
            next_state = choice(possible_next_states, 1, p=self.trans_prob)[0]
        else:
            # If current state is a terminal state (5-8), stay in the same state
            next_state = current_state
            
        self.human_state = next_state
        reward = self._get_reward(next_state)
        done = next_state >= self.output_states_offset
        
        if done:
            self.visited_goal_state = next_state
            
        return self.human_state, reward, done, {}

    def _get_reward(self, state):
        """Calculate reward for given state"""
        if state >= self.output_states_offset:
            return self.reward_map[state - self.output_states_offset]
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