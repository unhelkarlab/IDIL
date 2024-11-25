import gym
from gym.spaces import Discrete, Box
from .mental_iql import MentalIQL
from omegaconf import DictConfig


def make_miql_agent(config: DictConfig, env: gym.Env):

  latent_dim = config.dim_c
  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if not (isinstance(env.action_space, Discrete)
          or isinstance(env.action_space, Box)):
    raise RuntimeError(
        "Invalid action space: Only Discrete and Box action spaces supported")

  if isinstance(env.action_space, Discrete):
    action_dim = env.action_space.n
    discrete_act = True
  else:
    action_dim = env.action_space.shape[0]
    discrete_act = False

  agent = MentalIQL(config, obs_dim, action_dim, latent_dim, discrete_obs,
                    discrete_act)
  return agent
