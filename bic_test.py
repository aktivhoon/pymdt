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
    sarsa = SARSA(num_actions=env.NUM_ACTIONS, output_offset=env.output_states_offset,
                  reward_map=env.reward_map, learning_rate=lr)
    forward = FORWARD(num_states=env.num_states, num_actions=env.NUM_ACTIONS,
                      output_offset=env.output_states_offset, reward_map=env.reward_map,
                      learning_rate=lr)
    
    for _ in range(n_pretrain_episodes):
        state = env.reset()
        done = False
        while not done:
            Q_vals = sarsa.get_Q_values(state)
            action = int(np.argmax(Q_vals))
            next_state, reward, done, _ = env.step((MDP.HUMAN_AGENT_INDEX, action))
            reward = 0  
            next_Q_vals = sarsa.get_Q_values(next_state)
            next_action = int(np.argmax(next_Q_vals))
            sarsa.optimize(reward, action, next_action, state, next_state)
            forward.optimize(state, reward, action, next_state, env.is_flexible)
            state = next_state
    return sarsa, forward
 


def simulate_episode(episode_df, env, n_pretrain_episodes, arb_params, total_simul=20):
    """
    Simulate a single episode, running `total_simul` simulations in parallel.
    """
    """
    Run a single simulation of an episode.
    This is used to parallelize `total_simul` runs inside `simulate_episode`.
    """
    threshold, rl_lr, max_trans_rate_mb_to_mf, max_trans_rate_mf_to_mb, temperature, est_lr = arb_params

    # Pretrain agents
    sarsa_sim, forward_sim = pretrain_agents(n_pretrain_episodes, est_lr, env)

    # Initialize arbitrator
    arb = Arbitrator(AssocRelEstimator(rl_lr),
                     BayesRelEstimator(thereshold=threshold),
                     max_trans_rate_mb_to_mf=max_trans_rate_mb_to_mf,
                     max_trans_rate_mf_to_mb=max_trans_rate_mf_to_mb,
                     temperature=temperature)

    sim_nll = 0.0
    prev_goal = None

    for _, row in episode_df.iterrows():
        s1, s2, s3 = int(row['orig_S1']) - 1, int(row['orig_S2']) - 1, int(row['orig_S3']) - 1
        a1, a2 = int(row['A1']) - 1, int(row['A2']) - 1
        final_reward = float(row['reward'])
        is_flexible = int(row['goal_state']) == -1
        current_goal = row['goal_state']

        if prev_goal is None:
            if is_flexible:
                arb.p_mb = 0.2
                arb.p_mf = 0.8
            else:
                arb.p_mb = 0.8
                arb.p_mf = 0.2

        if prev_goal is not None and current_goal != prev_goal:
            try:
                forward_sim.backward_update(current_goal)
                if is_flexible:
                    arb.p_mb = 0.2
                    arb.p_mf = 0.8
                else:
                    arb.p_mb = 0.8
                    arb.p_mf = 0.2
            except AttributeError:
                pass
        prev_goal = current_goal  

        # --- First decision step ---
        mf_Q1, mb_Q1 = sarsa_sim.get_Q_values(s1), forward_sim.get_Q_values(s1)
        integrated_Q1 = arb.get_Q_values(mf_Q1, mb_Q1)
        exp_Q1 = np.exp(np.array(integrated_Q1) * temperature)
        policy1 = exp_Q1 / np.sum(exp_Q1)
        prob1 = max(policy1[a1], 1e-10)
        sim_nll += -np.log(prob1)

        spe1, rpe1 = forward_sim.optimize(s1, 0, a1, s2, is_flexible), sarsa_sim.optimize(0, a1, a2, s1, s2)
        arb.add_pe(rpe1, spe1)

        # --- Second decision step ---
        mf_Q2, mb_Q2 = sarsa_sim.get_Q_values(s2), forward_sim.get_Q_values(s2)
        integrated_Q2 = arb.get_Q_values(mf_Q2, mb_Q2)
        exp_Q2 = np.exp(np.array(integrated_Q2) * temperature)
        policy2 = exp_Q2 / np.sum(exp_Q2)
        prob2 = max(policy2[a2], 1e-10)
        sim_nll += -np.log(prob2)

        spe2 = forward_sim.optimize(s2, final_reward, a2, s3, is_flexible)
        rpe2 = sarsa_sim.optimize(final_reward, a2, 0, s2, s3) if is_flexible else sarsa_sim.optimize(max(final_reward, 0), a2, 0, s2, s3)
        arb.add_pe(rpe2, spe2)

    return sim_nll

def compute_neg_log_likelihood_arbitrator(params, param_names, behavior_df, env, n_pretrain_episodes, total_simul=50):
    """
    Compute total negative log likelihood over all episodes.
    Uses multiprocessing for episode-level simulations.
    """
    arb_params = params  # Tuple of parameters
    episodes = [episode_df for _, episode_df in behavior_df.groupby('block')]

    # Prepare arguments as tuples for starmap
    episode_args = [(ep, env, n_pretrain_episodes, arb_params, total_simul) for ep in episodes]

    with mp.Pool() as pool:
        results = pool.starmap(simulate_episode, episode_args)  # Use starmap instead of map
    return np.sum(results)


class ArbitratorModelFitter:
    def __init__(self, n_optimization_runs=50, n_pretrain_episodes=1, total_simul=50):
        self.n_optimization_runs = n_optimization_runs
        self.n_pretrain_episodes = n_pretrain_episodes
        self.total_simul = total_simul
        self.results = []

    def fit(self, behavior_df, env, param_bounds, param_names):
        """
        Fit the arbitrator model to behavioral data.
        Uses a for-loop (not multiprocessing) to track progress.
        """
        self.results = []
        best_params = None
        best_nll = float('inf')

        current_params = [0.5, 0.3, 1.0, 1.0, 0.2, 0.2]
        objective = lambda params: compute_neg_log_likelihood_arbitrator(
                params, param_names, behavior_df, env, self.n_pretrain_episodes, total_simul=self.total_simul)
        for i in tqdm(range(self.n_optimization_runs), desc="Optimization Progress"):
            #result = minimize(objective, initial_params, method='L-BFGS-B', bounds=param_bounds)
            result = minimize(objective, current_params, method='Nelder-Mead',
                              options={'maxiter': 3, 'disp': False})
            current_params = result.x
            self.results.append((result.x, result.fun))
            print(f"Run {i+1}/{self.n_optimization_runs}: NLL = {result.fun:.4f}")
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
            print(f"  Best NLL so far: {best_nll:.4f}")
            print(f"  Best params so far: {best_params}")
        
        self.best_result = ModelFitResult(
            parameters=dict(zip(param_names, best_params)),
            neg_log_likelihood=best_nll,
            bic_score=2 * best_nll + len(param_bounds) * np.log(len(behavior_df)),
            aic_score=2 * best_nll + 2 * len(param_bounds),
            parameter_bounds=param_bounds,
            parameter_names=param_names
        )
        return self.best_result

if __name__ == "__main__":
    behavior_df = pd.read_csv('./behav_data/SUB002_BHV.csv')
    behavior_df.columns = ['block', 'trial', 'block_setting', 'orig_S1', 'orig_S2', 'orig_S3',
                           'A1', 'A2', 'RT(A1)', 'RT(A2)', 'onset(S1)', 'onset(S2)', 'onset(S3)',
                           'onset(A1)', 'onset(A2)', 'reward', 'total_reward', 'goal_state']
    
    param_bounds = [(0.1, 0.9), (0.01, 0.9), (0.1, 10.0), (0.1, 10.0), (0.01, 1.0), (0.01, 0.9)]
    param_names = ['threshold', 'rl_learning_rate', 'max_trans_rate_mb_to_mf', 'max_trans_rate_mf_to_mb', 'temperature', 'estimator_learning_rate']
    
    env = MDP()
    
    fitter = ArbitratorModelFitter(n_optimization_runs=100, n_pretrain_episodes=2, total_simul=25)
    best_fit_result = fitter.fit(behavior_df, env, param_bounds, param_names)
    
    print("\nBEST PARAMETER SET FOUND:")
    print(best_fit_result)
