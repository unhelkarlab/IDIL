from typing import Type
import gym
from gym.spaces import Discrete, Box
from .option_models import (SoftDiscreteOptionActor, DiagGaussianOptionActor,
                            SoftDiscreteOptionThinker, OptionDoubleQCritic,
                            OptionSingleQCritic)
from .option_iql import OptionIQL, OptionSAC
from omegaconf import DictConfig


def make_oiql_agent(config: DictConfig, env: gym.Env):
  'discrete observation may not work well'

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
    actor = SoftDiscreteOptionActor(config, obs_dim, action_dim, latent_dim)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianOptionActor(config, obs_dim, action_dim, latent_dim)

  thinker = SoftDiscreteOptionThinker(config, obs_dim, action_dim, latent_dim)
  if config.iql_single_critic:
    critic = OptionSingleQCritic(config, obs_dim, action_dim, latent_dim)
  else:
    critic = OptionDoubleQCritic(config, obs_dim, action_dim, latent_dim)

  agent = OptionIQL(config, obs_dim, action_dim, latent_dim, discrete_obs,
                    critic, actor, thinker)

  return agent


def make_osac_agent(config: DictConfig, env: gym.Env):
  'discrete observation may not work well'

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
    actor = SoftDiscreteOptionActor(config, obs_dim, action_dim, latent_dim)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianOptionActor(config, obs_dim, action_dim, latent_dim)

  thinker = SoftDiscreteOptionThinker(config, obs_dim, action_dim, latent_dim)
  critic = OptionDoubleQCritic(config, obs_dim, action_dim, latent_dim)

  agent = OptionSAC(config, obs_dim, action_dim, latent_dim, discrete_obs,
                    critic, actor, thinker)

  return agent
