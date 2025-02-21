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
from fminsearchbnd import fminsearchbnd

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

def pretrain_agents(n_pretrain_episodes, env):
    sarsa = SARSA(num_actions=env.NUM_ACTIONS, output_offset=env.output_states_offset,
                  reward_map=env.reward_map, learning_rate=0.18)
    forward = FORWARD(num_states=env.num_states, num_actions=env.NUM_ACTIONS,
                      output_offset=env.output_states_offset, reward_map=env.reward_map,
                      learning_rate=0.15)
    
    for _ in range(n_pretrain_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(env.NUM_ACTIONS)
            next_state, reward, done, _ = env.step((MDP.HUMAN_AGENT_INDEX, action))
            next_action = np.random.choice(env.NUM_ACTIONS)
            pretrain_reward = 0
            sarsa.update(pretrain_reward, action, next_action, state, next_state)
            forward.update(state, reward, action, next_state, env.bwd_idf, pretrain=True)
            from collections import defaultdict
            T_values = defaultdict(lambda: np.zeros(env.NUM_ACTIONS))
            for s in range(5):
                T_values[s] = forward.T[s]
            # print(f"T_values: {T_values}")
            state = next_state
    return sarsa, forward
 
def pretrain_with_csv(env, temp=0.1):
    sarsa = SARSA(num_actions=env.NUM_ACTIONS, output_offset=env.output_states_offset,
                  reward_map=env.reward_map, learning_rate=0.18, beta=temp)
    forward = FORWARD(num_states=env.num_states, num_actions=env.NUM_ACTIONS,
                      output_offset=env.output_states_offset, reward_map=env.reward_map,
                      learning_rate=0.15, beta=temp)

    behavior_df = pd.read_csv('./behav_data/dep_behav1_pre.csv', header=None)
    behavior_df.columns = ['block', 'trial', 'block_setting', 'orig_S1', 'orig_S2', 'orig_S3',
                           'A1', 'A2', 'RT(A1)', 'RT(A2)', 'onset(S1)', 'onset(S2)', 'onset(S3)',
                           'onset(A1)', 'onset(A2)', 'reward', 'total_reward', 'goal_state']
    
    for _, row in behavior_df.iterrows():
        s1, s2, s3 = int(row['orig_S1']) - 1, int(row['orig_S2']) - 1, int(row['orig_S3']) - 1
        a1, a2 = int(row['A1']) - 1, int(row['A2']) - 1
        final_reward = float(row['reward'])
        is_flexible = int(row['goal_state']) == -1
        current_goal = row['goal_state']

        forward.update(s1, 0, a1, s2, current_goal)
        sarsa.update(s1, a1, 0, s2)

        forward.update(s2, final_reward, a2, s3, current_goal)
        sarsa.update(s2, a2, final_reward, s3)

    # print(f"Pretraining complete. Forward and SARSA agents are ready.")
    # print(f"Forward Q-values: {forward.Q_fwd}")
    # print(f"SARSA Q-values: {sarsa.Q_sarsa}")
    # import time
    # time.sleep(100)
    return sarsa, forward

def simulate_episode(episode_df, env, n_pretrain_episodes, arb_params):
    """
    Simulate a single episode, running `total_simul` simulations in parallel.
    """
    """
    Run a single simulation of an episode.
    This is used to parallelize `total_simul` runs inside `simulate_episode`.
    """
    threshold, rl_lr, max_trans_rate_mb_to_mf, max_trans_rate_mf_to_mb, temperature, est_lr = arb_params

    # Pretrain agents
    #sarsa_sim, forward_sim = pretrain_agents(n_pretrain_episodes, env)
    sarsa_sim, forward_sim = pretrain_with_csv(env, temperature)
    sarsa_sim.lr = est_lr
    forward_sim.lr = est_lr

    # Initialize arbitrator
    arb = Arbitrator(AssocRelEstimator(rl_lr),
                     BayesRelEstimator(thereshold=threshold),
                     max_trans_rate_mb_to_mf=max_trans_rate_mb_to_mf,
                     max_trans_rate_mf_to_mb=max_trans_rate_mf_to_mb,
                     temperature=temperature)

    sim_nll = 0.0
    
    prev_goal = None
    for ep in episode_df:
        for _, row in ep.iterrows():
            s1, s2, s3 = int(row['orig_S1']) - 1, int(row['orig_S2']) - 1, int(row['orig_S3']) - 1
            a1, a2 = int(row['A1']) - 1, int(row['A2']) - 1
            final_reward = float(row['reward'])
            is_flexible = int(row['goal_state']) == -1
            current_goal = int(row['goal_state'])
            if current_goal != -1:
                current_goal = int(current_goal) - 1
            backward_update = False

            if prev_goal is None:
                if is_flexible:
                    arb.p_mb = 0.2
                    arb.p_mf = 0.8
                else:
                    arb.p_mb = 0.8
                    arb.p_mf = 0.2

            if prev_goal is not None and current_goal != prev_goal:
                # print(f"The original was: {prev_goal} and the current is: {current_goal}")
                try:
                    backward_update = True
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
            if backward_update:
                forward_sim.backward_update(current_goal)
                # print(f"forward backward update: {forward_sim.Q_fwd}")
                # import time
                # time.sleep(100)
                # print(f"backward update")
            mf_Q1, mb_Q1 = sarsa_sim.get_Q_values(s1), forward_sim.get_Q_values(s1)
            integrated_Q1 = arb.get_Q_values(mf_Q1, mb_Q1)
            exp_Q1 = np.exp(np.array(integrated_Q1) * temperature)
            policy1 = exp_Q1 / np.sum(exp_Q1)
            prob1 = max(policy1[a1], 1e-10)
            sim_nll += -np.log(prob1)

            spe1, rpe1 = forward_sim.update(s1, 0, a1, s2, current_goal), sarsa_sim.update(s1, a1, 0, s2)
            arb.add_pe(rpe1, spe1)

            # --- Second decision step ---
            if backward_update:
                forward_sim.backward_update(current_goal)
                # print(f"backward update")
            mf_Q2, mb_Q2 = sarsa_sim.get_Q_values(s2), forward_sim.get_Q_values(s2)
            integrated_Q2 = arb.get_Q_values(mf_Q2, mb_Q2)
            exp_Q2 = np.exp(np.array(integrated_Q2) * temperature)
            policy2 = exp_Q2 / np.sum(exp_Q2)
            prob2 = max(policy2[a2], 1e-10)
            sim_nll += -np.log(prob2)

            spe2 = forward_sim.update(s2, final_reward, a2, s3, current_goal)
            rpe2 = sarsa_sim.update(s2, a2, final_reward, s3)
            arb.add_pe(rpe2, spe2)
    return sim_nll

def compute_neg_log_likelihood_arbitrator(params, param_names, behavior_df, env, n_pretrain_episodes):
    """
    Compute total negative log likelihood over all episodes.
    Uses multiprocessing for episode-level simulations.
    """
    arb_params = params  # Tuple of parameters
    
    # Find indices where block transitions from 16 to 1
    block_transitions = np.where((behavior_df['block'].values[:-1] == 16) & 
                            (behavior_df['block'].values[1:] == 1))[0]

    # Create episode start and end indices
    episode_starts = [0] + [idx + 1 for idx in block_transitions]
    episode_ends = block_transitions.tolist() + [len(behavior_df) - 1]

    # Split the dataframe into episodes
    episode_df = [behavior_df.iloc[start:end+1].reset_index(drop=True) 
                for start, end in zip(episode_starts, episode_ends)]
    
    results = simulate_episode(episode_df, env, n_pretrain_episodes, arb_params)
    return results
    


class ArbitratorModelFitter:
    def __init__(self, n_optimization_runs=10, n_pretrain_episodes=2, max_iter=50):
        self.n_optimization_runs = n_optimization_runs
        self.n_pretrain_episodes = n_pretrain_episodes
        self.max_iter = max_iter
        self.results = []

    def fit(self, behavior_df, env, param_bounds, param_names):
        """
        Fit the arbitrator model to behavioral data.
        Uses a for-loop (not multiprocessing) to track progress.
        """
        self.results = []
        best_params = None
        best_nll = float('inf')

        LB = [bound[0] for bound in param_bounds]
        UB = [bound[1] for bound in param_bounds]

        current_params = [np.random.randint(1, 21) * ((UB[j] - LB[j]) / 20) + LB[j] for j in range(len(LB))]
        objective = lambda params: compute_neg_log_likelihood_arbitrator(
                params, param_names, behavior_df, env, self.n_pretrain_episodes)

        for i in tqdm(range(self.n_optimization_runs), desc="Optimization Progress"):
            error_count = 0
            success = False
            while not success and error_count < 1000:
                try:
                    x_opt, fval, exitflag, res = fminsearchbnd(objective, current_params, LB, UB, 
                                                                options={'maxiter': self.max_iter, 'disp': False})
                    success = True
                except Exception as e:
                    error_count += 1
                    # Reinitialize parameters randomly within the bounds, similar to MATLAB's randi initialization.
                    current_params = [np.random.uniform(LB[j], UB[j]) for j in range(len(LB))]
                    print(f"Optimization error: {e}. Reinitializing parameters. Count: {error_count}")
            if not success:
                print("Failed to optimize after 1000 reinitializations. Moving on.")
    
            current_params = x_opt
            self.results.append((x_opt, fval))
            print(f"Run {i+1}/{self.n_optimization_runs}: NLL = {fval:.4f}")
            if fval < best_nll:
                best_nll = fval
                best_params = x_opt
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
    behavior_df = pd.read_csv('./behav_data/dep_behav1.csv')
    behavior_df.columns = ['block', 'trial', 'block_setting', 'orig_S1', 'orig_S2', 'orig_S3',
                           'A1', 'A2', 'RT(A1)', 'RT(A2)', 'onset(S1)', 'onset(S2)', 'onset(S3)',
                           'onset(A1)', 'onset(A2)', 'reward', 'total_reward', 'goal_state']
    
    param_bounds = [(0.1, 0.9), (0.01, 0.9), (0.1, 10.0), (0.1, 10.0), (0.01, 1.0), (0.01, 0.9)]
    param_names = ['threshold', 'rl_learning_rate', 'max_trans_rate_mb_to_mf', 'max_trans_rate_mf_to_mb', 'temperature', 'estimator_learning_rate']
    
    env = MDP()
    
    fitter = ArbitratorModelFitter(n_optimization_runs=40, n_pretrain_episodes=80)
    best_fit_result = fitter.fit(behavior_df, env, param_bounds, param_names)
    
    print("\nBEST PARAMETER SET FOUND:")
    print(best_fit_result)
