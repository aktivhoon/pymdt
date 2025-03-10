import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import multiprocessing as mp
from dataclasses import dataclass
import traceback
import time
import queue
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.live import Live
from rich.table import Table
from rich import box

from utils import simulate_episode
from mdp import MDP
from fminsearchbnd import fminsearchbnd
from mdp import MDP
from fminsearchbnd import fminsearchbnd

def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Name of the behavior data file: must have .csv extension, with _pre.csv as well for pretraining')
    parser.add_argument('--n_optimization_runs', type=int, default=10, help='Number of optimization runs')
    parser.add_argument('--n_pretrain_episodes', type=int, default=80, help='Number of pretraining episodes')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for optimization')
    parser.add_argument('--sim_runs', type=int, default=10, help='Number of simulations per episode')
    parser.add_argument('--pretrain_scenario', type=str, default='sim', choices=['csv', 'sim'], help='Pretraining scenario: csv or sim')
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

def compute_neg_log_likelihood_arbitrator(params, behavior_df, env, n_simul, pretrain_scenario, n_pretrain_episodes, filename=None):
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
        result = simulate_episode(episode_df, env, arb_params, pretrain_scenario, n_pretrain_episodes, filename)
        results.append(result)
    
    return sum(results)/n_simul

def run_single_optimization(args):
    """Worker function for a single optimization run"""
    run_id, behavior_df, env, param_bounds, sim_runs, pretrain_scenario, n_pretrain_episodes, max_iter, filename, progress_queue = args
    
    # Create a local console object for this process
    console = Console()
    
    LB = [bound[0] for bound in param_bounds]
    UB = [bound[1] for bound in param_bounds]
    
    # Initialize parameters randomly within bounds - Use different random seed for each run
    np.random.seed(int(time.time()) + run_id) 
    current_params = [np.random.uniform(LB[j], UB[j]) for j in range(len(LB))]
    
    # Progress reporting
    def update_progress(iteration, total_iters, current_nll):
        try:
            progress_queue.put({
                'run_id': run_id,
                'iteration': iteration,
                'total': total_iters,
                'nll': current_nll,
                'status': 'running'
            })
        except Exception as e:
            print(f"Run {run_id}: Progress reporting error: {e}")
    
    # Create a wrapper for the objective function to track iterations
    iter_count = [0]
    best_nll = [float('inf')]
    
    def objective_with_tracking(params):
        iter_count[0] += 1
        
        # Enforce hard limit on iterations
        if iter_count[0] > max_iter:
            # Return a penalty value to encourage termination
            return best_nll[0] * 1.1
            
        nll = compute_neg_log_likelihood_arbitrator(
            params, behavior_df, env, sim_runs, pretrain_scenario, n_pretrain_episodes, filename)
        
        if nll < best_nll[0]:
            best_nll[0] = nll
            
        # Update progress every few iterations to avoid overwhelming the queue
        if iter_count[0] % 5 == 0 or iter_count[0] == 1:  
            update_progress(iter_count[0], max_iter, best_nll[0])
            
        return nll
    
    error_count = 0
    success = False
    
    while not success and error_count < 1000:
        try:
            # Reset iteration counter for each attempt
            iter_count[0] = 0
            best_nll[0] = float('inf')
            
            # Signal start of optimization
            try:
                progress_queue.put({
                    'run_id': run_id,
                    'iteration': 0,
                    'total': max_iter,
                    'nll': float('inf'),
                    'status': 'starting'
                })
            except Exception:
                pass
            
            # Ensure strict enforcement of max_iter
            options = {
                'maxiter': max_iter,
                'disp': False,
            }
            x_opt, fval, exitflag, res = fminsearchbnd(objective_with_tracking, current_params, LB, UB, options=options)
            success = True
            
            # Signal completion with actual iteration count
            try:
                progress_queue.put({
                    'run_id': run_id,
                    'iteration': min(iter_count[0], max_iter),  # Cap at max_iter for display
                    'total': max_iter,
                    'nll': fval,
                    'status': 'completed'
                })
            except Exception as e:
                console.print(f"[bold red]Run {run_id}:[/bold red] Failed to report completion: {e}")
            
        except Exception as e:
            error_count += 1
            # Reinitialize parameters randomly within the bounds
            current_params = [np.random.uniform(LB[j], UB[j]) for j in range(len(LB))]
            
            try:
                progress_queue.put({
                    'run_id': run_id,
                    'status': 'error',
                    'error_msg': str(e),
                    'error_count': error_count
                })
            except Exception:
                pass
            
            console.print(f"[bold red]Run {run_id}:[/bold red] Error in optimization: {e}")
            traceback.print_exc()
    
    if not success:
        try:
            progress_queue.put({
                'run_id': run_id,
                'status': 'failed',
                'error_count': error_count
            })
        except Exception:
            pass
        console.print(f"[bold red]Run {run_id}:[/bold red] Failed to optimize after 1000 reinitializations.")
        return (None, float('inf'))
    
    return (x_opt, fval)


class ArbitratorModelFitter:
    def __init__(self, n_optimization_runs=10, n_pretrain_episodes=80, max_iter=200, sim_runs=10, pretrain_scenario='csv', filename=None):
        self.n_optimization_runs = n_optimization_runs
        self.n_pretrain_episodes = n_pretrain_episodes
        self.max_iter = max_iter
        self.sim_runs = sim_runs
        self.pretrain_scenario = pretrain_scenario
        self.filename = filename
        self.results = []
        self.console = Console()

    def _progress_monitor(self, progress_queue, n_runs, stop_event):
        """Monitor thread to update the rich progress display with iteration enforcement"""
        """Monitor thread to update the rich progress display"""
        # Set up the rich progress display with fixed column widths
        progress_table = Table(show_header=True, header_style="bold", box=box.ASCII2)
        progress_table.add_column("Run", width=8)
        progress_table.add_column("Progress", width=23)
        progress_table.add_column("Iterations", width=10)
        progress_table.add_column("Best NLL", width=12)
        progress_table.add_column("Status", width=10)
        
        # Track progress for each run
        run_progress = {}
        completed_runs = 0
        
        with Live(progress_table, refresh_per_second=2, console=self.console) as live:
            while not stop_event.is_set() or not progress_queue.empty():
                try:
                    # Try to get an update from the queue (non-blocking)
                    try:
                        update = progress_queue.get(block=False)
                        run_id = update.get('run_id')
                        
                        # Update the progress tracking for this run
                        if run_id not in run_progress:
                            run_progress[run_id] = {
                                'iteration': 0,
                                'total': self.max_iter,
                                'nll': float('inf'),
                                'status': 'waiting'
                            }
                        
                        # Update with the newest information
                        run_progress[run_id].update(update)
                        
                        # Check if a run just completed
                        if update.get('status') == 'completed' and run_progress[run_id].get('status') != 'completed':
                            completed_runs += 1
                    except queue.Empty:
                        # Queue is empty, just continue
                        pass
                    
                    # Rebuild the table (do this regardless of whether we got an update)
                    progress_table = Table(show_header=True, header_style="bold", box=box.ASCII2)
                    progress_table.add_column("Run", width=8)
                    progress_table.add_column("Progress", width=23)
                    progress_table.add_column("Iterations", width=10)
                    progress_table.add_column("Best NLL", width=12)
                    progress_table.add_column("Status", width=10)
                    
                    for rid in sorted(run_progress.keys()):
                        run_data = run_progress[rid]
                        
                        # FIX #2: Better progress bar with Rich formatting
                        if run_data['total'] > 0:
                            progress_pct = min(100, int(100 * run_data['iteration'] / run_data['total']))
                            bar_length = 20
                            filled_length = int(bar_length * progress_pct / 100)
                            # Use rich formatting for better visibility
                            progress_bar = f"[{'=' * filled_length}{' ' * (bar_length - filled_length)}]"
                        else:
                            progress_bar = "[                    ]"
                            
                        # Format iteration info with enforcement of limits for display
                        current_iter = min(run_data['iteration'], run_data['total'])
                        iter_info = f"{current_iter}/{run_data['total']}"
                        
                        # Format NLL
                        nll_str = f"{run_data['nll']:.6f}" if run_data['nll'] != float('inf') else "---"
                        
                        # Format status with color
                        status = run_data['status']
                        if status == 'completed':
                            status_str = "[green]Completed[/green]"
                        elif status == 'error':
                            status_str = f"[red]Error ({run_data.get('error_count', '?')})[/red]"
                        elif status == 'failed':
                            status_str = "[red]Failed[/red]"
                        else:
                            status_str = "[yellow]Running[/yellow]"
                        
                        # Add row to table with better formatting
                        progress_table.add_row(
                            f"Run {rid}",
                            f"[blue]{progress_bar}[/blue]",
                            iter_info,
                            nll_str,
                            status_str
                        )
                    
                    # Update the live display
                    live.update(progress_table)
                    
                except Exception as e:
                    self.console.print(f"[bold red]Error in progress monitor:[/bold red] {e}")
                    traceback.print_exc()
                    time.sleep(1)  # Pause longer if there's an error
        
        return completed_runs

    def fit(self, behavior_df, env, param_bounds, param_names):
        """
        Fit the arbitrator model to behavioral data.
        Parallelize the optimization runs, not the simulations.
        Uses rich for visual progress tracking.
        """
        self.results = []
        
        # Use a manager to create a queue that can be shared between processes
        manager = mp.Manager()
        progress_queue = manager.Queue()
        stop_event = manager.Event()
        
        # Prepare arguments for parallel optimization runs
        args_list = [
            (i, behavior_df, env, param_bounds, self.sim_runs, self.pretrain_scenario, 
             self.n_pretrain_episodes, self.max_iter, self.filename, progress_queue) 
            for i in range(self.n_optimization_runs)
        ]
        
        # Start the progress monitor in a separate thread
        import threading
        self.console.print("[bold blue]Initializing progress monitor...[/bold blue]")
        monitor_thread = threading.Thread(
            target=self._progress_monitor, 
            args=(progress_queue, self.n_optimization_runs, stop_event)
        )
        monitor_thread.daemon = True  # Make sure thread terminates when main program ends
        monitor_thread.start()
        time.sleep(0.5)  # Give the monitor thread time to initialize
        
        # Run optimizations in parallel
        self.console.print(f"[bold]Starting {self.n_optimization_runs} parallel optimization runs...[/bold]")
        with mp.Pool(processes=min(self.n_optimization_runs, mp.cpu_count())) as pool:
            self.results = pool.map(run_single_optimization, args_list)
        
        # Signal the monitor thread to stop and wait for it
        stop_event.set()
        monitor_thread.join()
        
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
    
    console = Console()
    console.print("[bold cyan]Arbitrator Model Fitting[/bold cyan]")
    console.print(f"[bold]Data file:[/bold] {args.filename}")
    console.print(f"[bold]Optimization runs:[/bold] {args.n_optimization_runs}")
    console.print(f"[bold]Simulation runs per evaluation:[/bold] {args.sim_runs}")
    console.print(f"[bold]Max iterations per optimization:[/bold] {args.max_iter}")
    
    fitter = ArbitratorModelFitter(n_optimization_runs=args.n_optimization_runs, 
                                   n_pretrain_episodes=args.n_pretrain_episodes, 
                                   max_iter=args.max_iter, 
                                   sim_runs=args.sim_runs,
                                   pretrain_scenario=args.pretrain_scenario,
                                   filename=args.filename
                                   )
    best_fit_result = fitter.fit(behavior_df, env, param_bounds, param_names)
    
    console.print("\n[bold green]BEST PARAMETER SET FOUND:[/bold green]")
    console.print(str(best_fit_result))

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
    
    console.print(f"[bold green]Results saved to: {args.output_dir}/{args.filename}_arbitrator_fit_results.csv[/bold green]")