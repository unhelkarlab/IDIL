from tqdm import tqdm
from idil_train import vis_cleanup as vis_utils
import os 
from collections import defaultdict
import numpy as np
import torch 
import utils as loading_utils
from idil_algs.IDIL.agent.mental_iql import MentalIQL
from idil_algs.baselines.IQLearn.utils.utils import make_env, eval_mode
from itertools import count
import idil_gym
import gym
from munch import Munch 

def load_agent(run_id: str, env_name:str ="CleanupSingle-v0"):
    """
    Load agent and environment
    """

    # load run config and agent
    run_path = loading_utils.get_run_path(env_name, run_id)
    run_config = loading_utils.get_run_config(run_path)
    agent = loading_utils.get_agent(run_path, run_config)

    return agent, run_config

def generate_trajectory(agent: MentalIQL, run_config: Munch, env):
    """
    Generate a single trajectory from the agent in the environment
    """
    # initialize environment
    state = env.reset()
    done = False
    prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    latent = agent.choose_mental_state(state, prev_lat, sample=True)
    reward = 0 # initialize reward

    # initialize data storage for trajs
    trajectory_data = defaultdict(list)

    for step in count():
        with eval_mode(agent):
            action = agent.choose_policy_action(state, latent, sample=True)
            next_state, _reward, done, info = env.step(action)
            next_latent = agent.choose_mental_state(next_state, latent, sample=True)
        
        reward += _reward

        # save data
        trajectory_data['states'].append(state)
        trajectory_data['actions'].append(action)
        trajectory_data['rewards'].append(_reward)
        trajectory_data['latents'].append(latent)
        trajectory_data['next_states'].append(next_state)
        trajectory_data['next_latents'].append(next_latent)

        # finish if done or hitting time limit
        if done or (step == int(run_config.max_explore_step)):
            return trajectory_data

        # update state and latent
        state = next_state
        latent = next_latent


def generate_agent_trajectories(run_id:str, env_name:str="CleanupSingle-v0", num_trajectories: int = 5):
    """
    Load agent and environment and generate `num_trajectories` trajectories
    """
    # load agent
    agent, config = load_agent(run_id, env_name)
    env = make_env(env_name)
    
    # generate trajectories
    trajectories = defaultdict(list)

    for _ in tqdm(range(num_trajectories)):
        traj = generate_trajectory(agent, config, env)
        for key, value in traj.items():
            trajectories[key].append(value)

    return trajectories
