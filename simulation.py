import numpy as np
import pandas as pd
from tqdm import tqdm

from mdp import MDP
from models import SARSA, FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator

# Constants
MDP_STAGES = 2
TOTAL_EPISODES = 100
TRIALS_PER_EPISODE = 80
COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'score']
DETAIL_COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'score', 
                 'rpe1', 'rpe2', 'spe1', 'spe2', 'action1', 'action2', 
                 'state1', 'state2', 'trans_prob', 'goal_state']

def compute_human_action(arbitrator, human_obs, model_free, model_based):
    """Compute human action by integrating model-free and model-based values"""
    return arbitrator.action(model_free.get_Q_values(human_obs),
                           model_based.get_Q_values(human_obs))

def simulation(threshold=BayesRelEstimator.THRESHOLD, 
              estimator_learning_rate=AssocRelEstimator.LEARNING_RATE,
              amp_mb_to_mf=Arbitrator.AMPLITUDE_MB_TO_MF, 
              amp_mf_to_mb=Arbitrator.AMPLITUDE_MF_TO_MB,
              temperature=Arbitrator.SOFTMAX_TEMPERATURE, 
              rl_learning_rate=0.2):
    """Run simulation with human agents using SARSA and FORWARD learning models.
    
    Args:
        threshold: Threshold for Bayesian relative estimator
        estimator_learning_rate: Learning rate for associative relative estimator
        amp_mb_to_mf: Amplitude of model-based to model-free influence 
        amp_mf_to_mb: Amplitude of model-free to model-based influence
        temperature: Temperature parameter for softmax action selection
        rl_learning_rate: Learning rate for SARSA and FORWARD models
        
    Returns:
        Tuple of DataFrames containing episode and trial-level results
    """
    # Initialize environment and data collection
    env = MDP(MDP_STAGES)
    res_data_df = pd.DataFrame(columns=COLUMNS)
    res_detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    
    # Initialize agents
    sarsa = SARSA(num_actions=env.NUM_ACTIONS,
                  output_offset=env.output_states_offset,
                  reward_map=env.reward_map,
                  learning_rate=rl_learning_rate)
    
    forward = FORWARD(num_states=env.num_states,
                     num_actions=env.NUM_ACTIONS,
                     output_offset=env.output_states_offset,
                     reward_map=env.reward_map,
                     learning_rate=rl_learning_rate)
    
    arb = Arbitrator(AssocRelEstimator(estimator_learning_rate),
                    BayesRelEstimator(thereshold=threshold),
                    amp_mb_to_mf=amp_mb_to_mf,
                    amp_mf_to_mb=amp_mf_to_mb,
                    temperature=temperature)
    
    # Run episodes
    for episode in tqdm(range(TOTAL_EPISODES)):
        cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_score = 0
        
        for trial in range(TRIALS_PER_EPISODE):
            t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_score = 0
            rpe1 = rpe2 = spe1 = spe2 = 0
            act1 = act2 = stt1 = stt2 = 0
            
            # Reset environment
            game_terminate = False
            human_obs = env.reset()
            current_game_step = 0

            forward.bwd_update(env.bwd_idf)
            
            # Run single trial
            while not game_terminate:
                # Human chooses and executes action
                human_action = compute_human_action(arb, human_obs, sarsa, forward)
                next_human_obs, human_reward, game_terminate, _ = env.step(
                    (MDP.HUMAN_AGENT_INDEX, human_action))
                
                # Update forward model
                spe = forward.optimize(human_obs, human_reward, human_action, 
                                    next_human_obs, env.is_flexible)
                
                # Get next action for SARSA update
                next_human_action = compute_human_action(arb, next_human_obs, 
                                                       sarsa, forward)
                
                # Update SARSA model
                if env.is_flexible:
                    rpe = sarsa.optimize(human_reward, human_action, next_human_action, 
                                       human_obs, next_human_obs)
                else:
                    # Normalize reward for specific goal condition
                    norm_reward = human_reward if human_reward > 0 else 0
                    rpe = sarsa.optimize(norm_reward, human_action, next_human_action, 
                                       human_obs, next_human_obs)
                
                # Update arbitrator
                mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
                
                # Accumulate trial statistics
                t_d_p_mb += d_p_mb
                t_p_mb += p_mb
                t_mf_rel += mf_rel
                t_mb_rel += mb_rel
                t_rpe += abs(rpe)
                t_spe += spe
                t_score += human_reward
                
                # Store step information
                human_obs = next_human_obs
                if current_game_step == 0:
                    rpe1, spe1, act1, stt1 = rpe, spe, human_action, human_obs
                else:
                    rpe2, spe2, act2, stt2 = rpe, spe, human_action, human_obs
                current_game_step += 1
            
            # Process trial results
            d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = map(
                lambda x: x / MDP_STAGES, 
                [t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])
            
            # Accumulate episode statistics
            cum_d_p_mb += d_p_mb
            cum_p_mb += p_mb
            cum_mf_rel += mf_rel
            cum_mb_rel += mb_rel
            cum_rpe += rpe
            cum_spe += spe
            cum_score += t_score
            
            # Store trial details
            detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_score,
                         rpe1, rpe2, spe1, spe2, act1, act2, stt1, stt2,
                         env.trans_prob[0], env.visited_goal_state]
            res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col
        
        # Store episode results
        episode_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                             [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel,
                              cum_p_mb, cum_d_p_mb, cum_score]))
        res_data_df.loc[episode] = episode_col
    
    return res_data_df, res_detail_df

if __name__ == "__main__":
    # Run simulation with default parameters
    episode_results, trial_results = simulation()
    
    print("\nEpisode Results:")
    print(episode_results.describe())
    
    print("\nTrial Results:")
    print(trial_results.describe())