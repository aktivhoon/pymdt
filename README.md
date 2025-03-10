# pymdt

Python Implementation of two step MDT, with fitting algorithm

## Current Status

### Forward Model

- [x] Confirmed that the forward model is working
- [x] Exactly same results as the MATLAB code

### SARSA Model

- [x] Confirmed that the SARSA model is working
- [x] Exactly same results as the MATLAB code (assuming the same random seed; not confident about the random seed)

### Arbitration

- [x] Confirmed that the arbitration model is working
- [x] Exactly same results as the MATLAB code for MB reliability estimation
- [x] Exactly same results as the MATLAB code for MF reliability estimation
- [x] Exactly same transition function as the MATLAB code

### Fitting

- [x] Confirmed that the fitting algorithm is working
- [x] Exactly same results as the MATLAB code

## Installation

Before running any scripts, install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: The code requires Python 3.7 or newer due to the use of dataclasses with type annotations.

## Parameter Fitting (param_fit.py)

The `param_fit.py` script implements a parallel optimization algorithm for fitting the arbitrator model to behavioral data. It uses multiple optimization runs to avoid local minima and find the best set of parameters.

### How It Works

1. The script loads behavioral data from CSV files and sets up the optimization environment.
2. It runs multiple optimization instances in parallel, each starting with different random initializations.
3. For each optimization run, it computes the negative log-likelihood of the model given the behavioral data.
4. It uses the Nelder-Mead method (via `fminsearchbnd`) to find parameter values that minimize the negative log-likelihood.
5. Progress for all runs is displayed in a live-updating table using the Rich library.
6. Once all runs complete, it selects the parameter set with the lowest negative log-likelihood as the best fit.
7. Results are saved to a CSV file in the specified output directory.

### Data Requirements

- Behavioral data should be saved in the `behav_data` directory in CSV format.
- The main data file should be named as specified in the `--filename` argument (e.g., `subject1.csv`).
- If using the 'csv' pretraining scenario, you need to provide a pretraining file with the same base name plus `_pre.csv` suffix (e.g., `subject1_pre.csv`).

### Command Line Arguments

```
usage: param_fit.py [-h] --filename FILENAME [--n_optimization_runs N_OPTIMIZATION_RUNS]
                     [--n_pretrain_episodes N_PRETRAIN_EPISODES] [--max_iter MAX_ITER]
                     [--sim_runs SIM_RUNS] [--pretrain_scenario PRETRAIN_SCENARIO]
                     [--output_dir OUTPUT_DIR]
```

- `--filename`: Name of the behavior data file (required, must have .csv extension)
- `--n_optimization_runs`: Number of parallel optimization runs to perform (default: 10)
- `--n_pretrain_episodes`: Number of pretraining episodes (default: 2)
- `--max_iter`: Maximum number of iterations for each optimization run (default: 200)
- `--sim_runs`: Number of simulations per likelihood evaluation (default: 10)
- `--pretrain_scenario`: Pretraining scenario, either 'csv' or 'agents' (default: 'csv')
- `--output_dir`: Directory where results will be saved (default: './results')

### Example Usage

```bash
python param_fit.py --filename subject1 --n_optimization_runs 20 --sim_runs 10 --max_iter 200 --output_dir ./results
```

This command will:

1. Load data from `./behav_data/subject1.csv`
2. Run 20 parallel optimization instances
3. Use 10 simulations per likelihood evaluation
4. Limit each optimization to 200 iterations
5. Save results to `./results/subject1_arbitrator_fit_results.csv`

### Output

The script produces a CSV file with:

- Command line arguments used
- Best parameter values found
- Negative log-likelihood, BIC, and AIC scores
- Parameter bounds used in the optimization

### Execution Example

When running the script, you'll see a live progress display similar to the one below:

![Parameter Fitting Progress Display](https://github.com/user-attachments/assets/6f84cac1-107f-4aca-9523-111dbe3f9497)

The table shows:

- Progress for each optimization run
- Current iteration count for each run
- Best negative log-likelihood (NLL) found so far
- Status of each run (Running, Completed, or Error)

This live display makes it easy to monitor the fitting process, especially for long-running optimizations with many parallel runs.