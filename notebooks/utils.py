import yaml 
from munch import Munch
import os
from idil_algs.baselines.IQLearn.utils.utils import make_env
from idil_algs.IDIL.agent.mental_iql import MentalIQL
import idil_gym
import gym

RESULTS_PATH="./idil_train/result/"


def get_run_path(env_name: str, run_id: str, results_path=RESULTS_PATH):
    """
    Get the path where we store 'model' and 'log' data for a given run
    """
    # read one dir below, as there is always a date directory
    _path = os.path.join(results_path, env_name, 'idil', run_id)
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

def get_agent(run_path: str, run_conf: Munch):
    """
    Load the agent from the run path
    """
    # load env
    env = make_env(run_conf.env_name)

    _obs_space_dim = env.observation_space.n if isinstance(env.observation_space, gym.spaces.Discrete) else env.observation_space.shape[0]
    _act_space_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]    

    miql_agent = MentalIQL(config=run_conf,
                           obs_dim=_obs_space_dim,
                           action_dim=_act_space_dim,
                           lat_dim=run_conf.dim_c, # obs dim and action dim are hardcoded for now, they belogn to CleanupSingle
                           discrete_obs=isinstance(env.observation_space, gym.spaces.Discrete),
                           discrete_act=isinstance(env.action_space, gym.spaces.Discrete))

    prefix = os.listdir(os.path.join(run_path, 'model'))[0].replace('_pi', '').replace('_tx', '').strip()

    miql_agent.load(os.path.join(run_path, 'model', prefix))
    return miql_agent