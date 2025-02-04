import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass

from mdp import MDP
from sarsa import SARSA
from forward import FORWARD
from arbitrator import Arbitrator, BayesRelEstimator, AssocRelEstimator

@dataclass
class ModelFitResult:
    parameters: dict
    neg_log_likelihood: float
    bic_score: float
    aic_score: float
    parameter_bounds: list
    parameter_names: list

    def __str__(self):
        lines = ["Arbitrator Model Fitting Results:"]
        lines.append("\nParameters:")
        for name, value in self.parameters.items():
            lines.append(f"  {name}: {value:.4f}")
        lines.append(f"\nNegative Log Likelihood: {self.neg_log_likelihood:.4f}")
        lines.append(f"BIC Score: {self.bic_score:.4f}")
        lines.append(f"AIC Score: {self.aic_score:.4f}")
        return "\n".join(lines)

def pretrain_agents(n_pretrain_episodes, lr, env):
    """Pretrain SARSA and FORWARD agents using zero rewards to learn transitions."""
    
    # Initialize agents
    sarsa = SARSA(num_actions=env.NUM_ACTIONS,
                  output_offset=env.output_states_offset,
                  reward_map=env.reward_map,
                  learning_rate=lr)
    
    forward = FORWARD(num_states=env.num_states,
                     num_actions=env.NUM_ACTIONS,
                     output_offset=env.output_states_offset,
                     reward_map=env.reward_map,
                     learning_rate=lr)
    
    for _ in range(n_pretrain_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Choose action using current Q-values
            Q_vals = sarsa.get_Q_values(state)
            action = int(np.argmax(Q_vals))
            
            # Take action and observe next state
            next_state, reward, done, _ = env.step((MDP.HUMAN_AGENT_INDEX, action))
            
            # Set reward to zero for pretraining
            reward = 0
            
            # Get next action for SARSA update
            next_Q_vals = sarsa.get_Q_values(next_state)
            next_action = int(np.argmax(next_Q_vals))
            
            # Update both models with zero reward
            sarsa.optimize(reward, action, next_action, state, next_state)
            forward.optimize(state, reward, action, next_state, env.is_flexible)
            
            state = next_state
            
    return sarsa, forward

def compute_neg_log_likelihood_arbitrator(params, param_names, behavior_df, env, n_pretrain_episodes):
    """Compute negative log likelihood for given arbitrator parameters, processing each episode separately."""
    threshold, rl_lr, amp_mb_to_mf, amp_mf_to_mb, temperature, est_lr = params
    total_nll = 0.0
    
    # Split data into episodes based on block numbers
    episodes = []
    current_episode = []
    prev_block = behavior_df.iloc[0]['block']
    
    for _, row in behavior_df.iterrows():
        if row['block'] != prev_block:
            episodes.append(pd.DataFrame(current_episode))
            current_episode = []
            prev_block = row['block']
        current_episode.append(row)
    
    if current_episode:
        episodes.append(pd.DataFrame(current_episode))
    
    # Process each episode separately
    for episode_df in episodes:
        # Reset and pretrain agents for each episode
        sarsa_sim, forward_sim = pretrain_agents(n_pretrain_episodes, est_lr, env)
        
        # Initialize arbitrator for this episode
        arb = Arbitrator(AssocRelEstimator(rl_lr),
                        BayesRelEstimator(thereshold=threshold),
                        amp_mb_to_mf=amp_mb_to_mf,
                        amp_mf_to_mb=amp_mf_to_mb,
                        temperature=temperature)
        
        # Process each trial in the episode
        for _, row in episode_df.iterrows():
            # Extract states and actions
            s1, s2, s3 = int(row['orig_S1'])-1, int(row['orig_S2'])-1, int(row['orig_S3'])-1
            a1, a2 = int(row['A1'])-1, int(row['A2'])-1
            final_reward = float(row['reward'])
            is_flexible = int(row['block_setting']) == 1
            
            # First step: S1 -> S2
            mf_Q1 = sarsa_sim.get_Q_values(s1)
            mb_Q1 = forward_sim.get_Q_values(s1)
            integrated_Q1 = arb.get_Q_values(mf_Q1, mb_Q1)
            exp_Q1 = np.exp(np.array(integrated_Q1) * temperature)
            policy1 = exp_Q1 / np.sum(exp_Q1)
            prob1 = max(policy1[a1], 1e-10)
            total_nll += -np.log(prob1)
            
            # Update models for first step
            spe1 = forward_sim.optimize(s1, 0, a1, s2, is_flexible)
            rpe1 = sarsa_sim.optimize(0, a1, a2, s1, s2)
            arb.add_pe(rpe1, spe1)
            
            # Second step: S2 -> S3
            mf_Q2 = sarsa_sim.get_Q_values(s2)
            mb_Q2 = forward_sim.get_Q_values(s2)
            integrated_Q2 = arb.get_Q_values(mf_Q2, mb_Q2)
            exp_Q2 = np.exp(np.array(integrated_Q2) * temperature)
            policy2 = exp_Q2 / np.sum(exp_Q2)
            prob2 = max(policy2[a2], 1e-10)
            total_nll += -np.log(prob2)
            
            # Update models for second step
            spe2 = forward_sim.optimize(s2, final_reward, a2, s3, is_flexible)
            if is_flexible:
                rpe2 = sarsa_sim.optimize(final_reward, a2, 0, s2, s3)
            else:
                norm_reward = final_reward if final_reward > 0 else 0
                rpe2 = sarsa_sim.optimize(norm_reward, a2, 0, s2, s3)
            
            arb.add_pe(rpe2, spe2)
    
    return total_nll

def optimization_run_arbitrator(args):
    """Single optimization run with random initial parameters."""
    initial_params, param_names, behavior_df, n_pretrain_episodes, bounds, env = args
    
    objective = lambda params: compute_neg_log_likelihood_arbitrator(
        params, param_names, behavior_df, env, n_pretrain_episodes)
    
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    return result.x, result.fun

class ArbitratorModelFitter:
    def __init__(self, n_optimization_runs=20, n_pretrain_episodes=10):
        self.n_optimization_runs = n_optimization_runs
        self.n_pretrain_episodes = n_pretrain_episodes
        self.result = None

    def fit(self, behavior_df, env, param_bounds, param_names):
        """Fit the arbitrator model to behavioral data."""
        # Generate random initial parameters
        initial_params_list = [
            [np.random.uniform(low, high) for (low, high) in param_bounds]
            for _ in range(self.n_optimization_runs)
        ]
        
        # Prepare arguments for parallel optimization
        args_list = [
            (params, param_names, behavior_df, self.n_pretrain_episodes, param_bounds, env)
            for params in initial_params_list
        ]
        
        # Run parallel optimization
        results = []
        with mp.Pool() as pool:
            for params, nll in tqdm(pool.imap_unordered(optimization_run_arbitrator, args_list),
                                  total=self.n_optimization_runs,
                                  desc="Optimizing Arbitrator"):
                results.append((params, nll))
        
        # Find best parameters
        best_params, best_nll = min(results, key=lambda r: r[1])
        
        # Store results
        self.result = ModelFitResult(
            parameters=dict(zip(param_names, best_params)),
            neg_log_likelihood=best_nll,
            bic_score=2 * best_nll + len(param_bounds) * np.log(len(behavior_df)),
            aic_score=2 * best_nll + 2 * len(param_bounds),
            parameter_bounds=param_bounds,
            parameter_names=param_names
        )
        return self.result

if __name__ == "__main__":
    # Example usage
    # Read and preprocess behavioral data
    behavior_df = pd.read_csv('../behav_data/SUB001_BHV.csv')
    behavior_df.columns = ['block', 'trial', 'block_setting', 'orig_S1', 'orig_S2', 'orig_S3',
                          'A1', 'A2', 'RT(A1)', 'RT(A2)', 'onset(S1)', 'onset(S2)', 'onset(S3)',
                          'onset(A1)', 'onset(A2)', 'reward', 'total_reward', 'goal_state']
    
    param_bounds = [
        (0.1, 0.9),  # threshold
        (0.01, 0.9), # rl_learning_rate
        (0.1, 10.0), # amp_mb_to_mf
        (0.1, 10.0),   # amp_mf_to_mb
        (0.01, 1.0), # temperature
        (0.01, 0.9), # estimator_learning_rate
    ]
    
    param_names = [
        'threshold', 'rl_learning_rate', 'amp_mb_to_mf', 'amp_mf_to_mb', 
        'temperature', 'estimator_learning_rate', 
    ]
    
    env = MDP()
    fitter = ArbitratorModelFitter(n_optimization_runs=200)
    fit_result = fitter.fit(behavior_df, env, param_bounds, param_names)
    print(fit_result)