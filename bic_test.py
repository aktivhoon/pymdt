import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass
import traceback

from utils import simulate_episode
from mdp import MDP
from fminsearchbnd import fminsearchbnd

def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Name of the behavior data file: must have .csv extension, with _pre.csv as well for pretraining')
    parser.add_argument('--n_optimization_runs', type=int, default=10, help='Number of optimization runs')
    parser.add_argument('--n_pretrain_episodes', type=int, default=2, help='Number of pretraining episodes')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for optimization')
    parser.add_argument('--sim_runs', type=int, default=10, help='Number of simulations per episode')
    parser.add_argument('--pretrain_scenario', type=str, default='csv', help='Pretraining scenario: csv or agents')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')

    return parser

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

def compute_neg_log_likelihood_arbitrator(params, behavior_df, env, n_simul, pretrain_scenario, n_pretrain_episodes):
    """
    Compute total negative log likelihood over all episodes.
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
    
    # Run simulations sequentially instead of in parallel
    results = []
    for _ in range(n_simul):
        result = simulate_episode(episode_df, env, arb_params, pretrain_scenario, n_pretrain_episodes)
        results.append(result)
    
    return sum(results)/n_simul

def run_single_optimization(args):
    """Worker function for a single optimization run"""
    run_id, behavior_df, env, param_bounds, sim_runs, pretrain_scenario, n_pretrain_episodes, max_iter = args
    
    LB = [bound[0] for bound in param_bounds]
    UB = [bound[1] for bound in param_bounds]
    
    # Initialize parameters randomly within bounds
    current_params = [np.random.uniform(LB[j], UB[j]) for j in range(len(LB))]
    
    objective = lambda params: compute_neg_log_likelihood_arbitrator(
            params, behavior_df, env, sim_runs, pretrain_scenario, n_pretrain_episodes)
    
    error_count = 0
    success = False
    
    while not success and error_count < 1000:
        try:
            x_opt, fval, exitflag, res = fminsearchbnd(objective, current_params, LB, UB, 
                                                      options={'maxiter': max_iter, 'disp': False})
            success = True
        except Exception as e:
            error_count += 1
            # Reinitialize parameters randomly within the bounds
            current_params = [np.random.uniform(LB[j], UB[j]) for j in range(len(LB))]
            print(f"Optimization run {run_id} error: {e}. Reinitializing parameters. Count: {error_count}")
            traceback.print_exc()
    
    if not success:
        print(f"Run {run_id}: Failed to optimize after 1000 reinitializations.")
        return (None, float('inf'))
    
    print(f"Run {run_id}: NLL = {fval:.8f}")
    return (x_opt, fval)


class ArbitratorModelFitter:
    def __init__(self, n_optimization_runs=10, n_pretrain_episodes=80, max_iter=200, sim_runs=10, pretrain_scenario='csv'):
        self.n_optimization_runs = n_optimization_runs
        self.n_pretrain_episodes = n_pretrain_episodes
        self.max_iter = max_iter
        self.sim_runs = sim_runs
        self.pretrain_scenario = pretrain_scenario
        self.results = []

    def fit(self, behavior_df, env, param_bounds, param_names):
        """
        Fit the arbitrator model to behavioral data.
        Parallelize the optimization runs, not the simulations.
        """
        self.results = []
        
        # Prepare arguments for parallel optimization runs
        args_list = [
            (i, behavior_df, env, param_bounds, self.sim_runs, self.pretrain_scenario, self.n_pretrain_episodes, self.max_iter) 
            for i in range(self.n_optimization_runs)
        ]
        
        # Run optimizations in parallel
        print(f"Starting {self.n_optimization_runs} parallel optimization runs...")
        with mp.Pool(processes=min(self.n_optimization_runs, mp.cpu_count())) as pool:
            self.results = list(tqdm(pool.imap(run_single_optimization, args_list), 
                                    total=self.n_optimization_runs, 
                                    desc="Optimization Progress"))
        
        # Find the best result
        valid_results = [(x, f) for x, f in self.results if x is not None]
        if not valid_results:
            raise RuntimeError("All optimization runs failed")
        
        best_params, best_nll = min(valid_results, key=lambda x: x[1])
        
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
    args = argparser().parse_args()

    behavior_df = pd.read_csv(f'./behav_data/{args.filename}.csv', header=None)
    behavior_df.columns = ['block', 'trial', 'block_setting', 'orig_S1', 'orig_S2', 'orig_S3',
                           'A1', 'A2', 'RT(A1)', 'RT(A2)', 'onset(S1)', 'onset(S2)', 'onset(S3)',
                           'onset(A1)', 'onset(A2)', 'reward', 'total_reward', 'goal_state']
    
    # param_bounds = [(0.1, 0.9), (0.01, 0.9), (0.1, 10.0), (0.1, 10.0), (0.01, 1.0), (0.01, 0.9)]
    param_bounds = [(0.3, 0.8), (0.01, 0.35), (0.1, 10.0), (0.1, 10.0), (0.01, 0.5), (0.01, 0.2)]
    param_names = ['threshold', 'rl_learning_rate', 'max_trans_rate_mb_to_mf', 'max_trans_rate_mf_to_mb', 'temperature', 'estimator_learning_rate']
    
    env = MDP()
    
    fitter = ArbitratorModelFitter(n_optimization_runs=args.n_optimization_runs, 
                                   n_pretrain_episodes=args.n_pretrain_episodes, 
                                   max_iter=args.max_iter, 
                                   sim_runs=args.sim_runs,
                                   pretrain_scenario=args.pretrain_scenario
                                   )
    best_fit_result = fitter.fit(behavior_df, env, param_bounds, param_names)
    
    print("\nBEST PARAMETER SET FOUND:")
    print(best_fit_result)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(f"{args.output_dir}/{args.filename}_arbitrator_fit_results.csv", 'w') as f:
        f.write("Argparse Arguments:\n")
        f.write(str(args))
        f.write("\n")
        f.write("\n")
        f.write(str(best_fit_result))
        f.write("\n")
        f.write("Parameter Bounds:\n")
        f.write(f"{param_names}\n")
        f.write(f"{param_bounds}\n")
        f.write("\n")
        f.write("Parameter Values:\n")
        f.write(f"{best_fit_result.parameters}\n")
        f.write("\n")
        f.write(f"Negative Log Likelihood: {best_fit_result.neg_log_likelihood:.4f}\n")
        f.write(f"BIC Score: {best_fit_result.bic_score:.4f}\n")
        f.write(f"AIC Score: {best_fit_result.aic_score:.4f}\n")
        f.write("\n")