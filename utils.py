import numpy as np
import pandas as pd

from mdp import MDP
from sarsa import SARSA
from forward import FORWARD
from arbitrator import Arbitrator, BayesRelEstimator, AssocRelEstimator

def pretrain_agents(n_pretrain_episodes, env):
    # model learning rate for forward and sarsa are fixed during pretraining
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
            sarsa.update(state, action, reward, next_state)
            current_goal = -1
            forward.update(state, reward, action, next_state, current_goal, pretrain=True)
            from collections import defaultdict
            T_values = defaultdict(lambda: np.zeros(env.NUM_ACTIONS))
            for s in range(5):
                T_values[s] = forward.T[s]
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
        current_goal = row['goal_state']

        forward.update(s1, 0, a1, s2, current_goal, pretrain=True)
        sarsa.update(s1, a1, 0, s2)

        forward.update(s2, final_reward, a2, s3, current_goal, pretrain=True)
        sarsa.update(s2, a2, final_reward, s3, -1)

    return sarsa, forward

def simulate_episode(episode_df, env, arb_params, pretrain_scenario='csv', n_pretrain_episodes=80):
    """
    Simulate a single episode, running `total_simul` simulations in parallel.
    """
    """
    Run a single simulation of an episode.
    This is used to parallelize `total_simul` runs inside `simulate_episode`.
    """
    threshold, rl_lr, max_trans_rate_mb_to_mf, max_trans_rate_mf_to_mb, temperature, est_lr = arb_params

    # Pretrain agents
    if pretrain_scenario == 'csv':
        sarsa_sim, forward_sim = pretrain_with_csv(env, temperature)
    else:
        sarsa_sim, forward_sim = pretrain_agents(n_pretrain_episodes, env)

    # Set learning rates
    sarsa_sim.learning_rate = est_lr
    forward_sim.learning_rate = est_lr

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
            current_goal = int(row['goal_state'])
            if current_goal != -1:
                current_goal = int(current_goal) - 1
            backward_update = False

            if current_goal != prev_goal:
                try:
                    if current_goal == -1:
                        arb.p_mb = 0.2
                        arb.p_mf = 0.8
                    else:
                        backward_update = True
                        arb.p_mb = 0.8
                        arb.p_mf = 0.2
                except AttributeError:
                    pass
            prev_goal = current_goal

            # --- First decision step ---
            if backward_update:
                forward_sim.backward_update(current_goal)
            mf_Q1, mb_Q1 = sarsa_sim.get_Q_values(s1), forward_sim.get_Q_values(s1)
            integrated_Q1 = arb.get_Q_values(mf_Q1, mb_Q1)
            exp_Q1 = np.exp(np.array(integrated_Q1) * temperature)
            policy1 = exp_Q1 / np.sum(exp_Q1)

            prob1 = policy1[a1]
            sim_nll += -np.log(prob1)

            spe1, rpe1 = forward_sim.update(s1, 0, a1, s2, current_goal), sarsa_sim.update(s1, a1, 0, s2)
            arb.add_pe(rpe1, spe1)
            
            # --- Second decision step ---
            if backward_update:
                forward_sim.backward_update(current_goal)
            mf_Q2, mb_Q2 = sarsa_sim.get_Q_values(s2), forward_sim.get_Q_values(s2)
            integrated_Q2 = arb.get_Q_values(mf_Q2, mb_Q2)
            exp_Q2 = np.exp(np.array(integrated_Q2) * temperature)
            policy2 = exp_Q2 / np.sum(exp_Q2)

            prob2 = policy2[a2]
            sim_nll += -np.log(prob2)

            spe2 = forward_sim.update(s2, final_reward, a2, s3, current_goal)
            rpe2 = sarsa_sim.update(s2, a2, final_reward, s3, -1)
            arb.add_pe(rpe2, spe2)
    return sim_nll