from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml 
from munch import Munch
import os
from idil_algs.baselines.IQLearn.utils.utils import make_env
from idil_algs.IDIL.agent.mental_iql import MentalIQL
import idil_gym
import gym
import generate_trajectories as traj_utils
from idil_algs.baselines.IQLearn.dataset.expert_dataset import ExpertDataset


RESULTS_PATH="./idil_train/result/"


def get_run_path(env_name: str, run_id: str, results_path=RESULTS_PATH, alg_name='idil'):
    """
    Get the path where we store 'model' and 'log' data for a given run
    """
    # read one dir below, as there is always a date directory
    _path = os.path.join(results_path, env_name, alg_name, run_id)
    _date_folder = os.listdir(_path)[0]
    return os.path.join(_path, _date_folder)

def get_run_config(run_path:str):
    """
    Parse run YAML configuration and return as Munch dictioanry object
    """

    with open(os.path.join(run_path, 'log', 'config.yaml') , "r") as f:
        run_conf = yaml.load(f, Loader=yaml.FullLoader)
        run_conf = Munch(run_conf)
    return run_conf

def get_agent(run_path: str, run_conf: Munch, load_micro: bool = True, expert_dataset: ExpertDataset = None):
    """
    Load the agent from the run path
    """
    # load env
    env = make_env(run_conf.env_name)

    _obs_space_dim = env.observation_space.n if isinstance(env.observation_space, gym.spaces.Discrete) else env.observation_space.shape[0]
    _act_space_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]    

    _use_fixed_pi = not load_micro

    miql_agent = MentalIQL(config=run_conf,
                           obs_dim=_obs_space_dim,
                           action_dim=_act_space_dim,
                           lat_dim=run_conf.dim_c, # obs dim and action dim are hardcoded for now, they belogn to CleanupSingle
                           discrete_obs=isinstance(env.observation_space, gym.spaces.Discrete),
                           discrete_act=isinstance(env.action_space, gym.spaces.Discrete),
                           fixed_pi=_use_fixed_pi, expert_dataset=expert_dataset)

    prefix = os.listdir(os.path.join(run_path, 'model'))[0].replace('_pi', '').replace('_tx', '').strip()

    miql_agent.load(os.path.join(run_path, 'model', prefix), load_micro=load_micro)
    return miql_agent


def get_trained_models(runs: list, env_name:str, load_micro: bool = True, expert_dataset: ExpertDataset = None):
    """
    Load models from a given list of run IDs (from wandb).
    Returns:
        - dictionary object where keys are the kval used to select top-K trajectories
    """

    agents = defaultdict(list)

    for run in runs:
        # load run path
        run_path = get_run_path(env_name=env_name, run_id=run)
        run_conf = get_run_config(run_path)
        agent = get_agent(run_conf=run_conf, run_path=run_path, load_micro=load_micro, expert_dataset=expert_dataset)

        # parse kval
        kval = int(run_conf.k * 100)

        agents[kval].append(agent)

    return agents

from tqdm import tqdm

def get_agent_trajectories(runs: list, env_name: str, num_trajectories: int = 10, load_micro: bool = True, expert_dataset: ExpertDataset = None):
    # generate 10 trajectories for each trained agent
    trajectories = defaultdict(list)

    for run in tqdm(runs):
        traj_data = traj_utils.generate_agent_trajectories(run_id=run, num_trajectories=num_trajectories, env_name=env_name, 
                                                           load_micro=load_micro, expert_dataset=expert_dataset)
        kval = int(run.split("-")[1])

        trajectories[kval].append(traj_data)

    return trajectories

def get_cumulative_rewards(trajectories: dict, agents: dict = None):
    """
    Calculate cumulative rewards for each trajectory
    """
    # compute cumulative rewards for each trajectorie and get a distribution of the cumulative rewards
    cum_rewards = defaultdict(list)

    for kval in trajectories:
        for agent_data in trajectories[kval]:
            rewards = agent_data["rewards"]
            for rew_array in rewards:
                cum_rewards[kval].append(np.sum(rew_array))
            

    # sanity check printing shapes of each inner element
    print(f"There are {len(cum_rewards)} kvals")
    if agents:
        _key = list(agents.keys())[0]
        print(f"Each kval has {len(cum_rewards[100])} cum_rewards, matching the {len(agents[_key])} x 10 trajectories generated for each agent (using key = {_key})")

    return cum_rewards


def plot_cum_rewards(cum_rewards: dict):
    _, ax = plt.subplots(1,1)

    data = [cum_rewards[kval] for kval in cum_rewards]

    ax.boxplot(data, labels=[f"k={kval}%" + r" of $\mathcal{X}'$" for kval in cum_rewards.keys()])

    ax.set_title("Distribution of cumulative rewards for generated trajectories")
    ax.set_xlabel("k-value")
    ax.set_ylabel("Cumulative reward")
    ax.grid(alpha=0.3)


def backtest_action_trajectory(expert_dataset, agent):
    """
    Given a precomputed set of latents and states,
    run the agent to sample an action and see how the actions differ
    """

    agent_action_trajs = []

    for traj_idx in range(len(expert_dataset.trajectories["states"])):
        traj_states = expert_dataset.trajectories["states"][traj_idx]
        traj_latents = expert_dataset.trajectories["latents"][traj_idx]

        _action_traj = []
        for _state, _lat in zip(traj_states, traj_latents):
            _action = agent.choose_policy_action(_state, _lat)
            _action_traj.append(_action)

        agent_action_trajs.append(_action_traj)

    return agent_action_trajs

def compute_action_accuracy(expert_dataset, agent):
    """
    Compute the action accuracy between expert and agent
    """

    agent_action_trajs = backtest_action_trajectory(expert_dataset, agent)

    accs = []
    for i in range(len(expert_dataset.trajectories["states"])):
        _test_acts_expert = np.array(expert_dataset.trajectories["actions"][i])
        _test_acts_agent =  np.array(agent_action_trajs[i])

        accs.append(np.sum(_test_acts_expert == _test_acts_agent) / len(_test_acts_expert))

    return accs

def compute_sequence_accuracy(pred_latents, true_latents):
    pred_latents = np.array(pred_latents)
    true_latents = np.array(true_latents)
    return np.sum(pred_latents == true_latents) / len(true_latents)


def compute_accuracy_by_kval(test_data, agents):
    """
    Compute the accuracy of the inferred latents for each agent
    for each (state, action) pair in the test data
    """
    accs_by_k = defaultdict(list)

    for states, actions, latents in tqdm(zip(test_data["states"], test_data["actions"], test_data["latents"])):
        for kval, agent_list in agents.items():
            for agent in agent_list:
                inferred_latents, _, _ = agent.infer_mental_states(states, actions)
                acc = compute_sequence_accuracy(inferred_latents.squeeze(1), latents)
                accs_by_k[kval].append(acc)

    return accs_by_k

def plot_sequence_accuracy_boxplot(accuracy_by_kval:dict):
    # plot comparison of accuracy distribution for each k
    _df = pd.DataFrame(accuracy_by_kval)
    _, ax = plt.subplots(1,1)

    _df.boxplot(ax=ax)
    ax.set_title(r"Sequence accuracy for inferred latents on test $\mathcal{D}$")
    # format y-axis to be percentage
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel(r"$\mathcal{X}'$ percentage")
