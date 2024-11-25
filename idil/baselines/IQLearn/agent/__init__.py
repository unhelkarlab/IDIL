import gym
from gym.spaces import Discrete, MultiDiscrete, Box
from .sac_models import DiscreteActor, DiagGaussianActor, SoftDiscreteActor
from .sac import SAC
from .sac_discrete import SAC_Discrete
from .softq import SoftQ
from .softq_models import SimpleQNetwork, SingleQCriticDiscrete
from .sac_models import DoubleQCritic, SingleQCritic
from omegaconf import DictConfig


def make_softq_agent(config: DictConfig, env: gym.Env):
  q_net_base = SimpleQNetwork

  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if not isinstance(env.action_space, Discrete):
    raise RuntimeError(
        "Invalid action space: only discrete action is supported")

  action_dim = env.action_space.n

  obs_dim = int(obs_dim)
  action_dim = int(action_dim)
  agent = SoftQ(config, obs_dim, action_dim, discrete_obs, q_net_base)

  return agent


def make_sac_agent(config: DictConfig, env: gym.Env):
  'discrete observation may not work well'

  if config.iql_single_critic:
    critic_base = SingleQCritic
  else:
    critic_base = DoubleQCritic
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
    actor = SoftDiscreteActor(obs_dim, action_dim, config.hidden_policy,
                              config.gumbel_temperature)
  else:
    action_dim = env.action_space.shape[0]
    actor = DiagGaussianActor(obs_dim, action_dim, config.hidden_policy,
                              config.log_std_bounds, config.bounded_actor,
                              config.use_nn_logstd, config.clamp_action_logstd)

  agent = SAC(config, obs_dim, action_dim, discrete_obs, critic_base, actor)

  return agent


def make_sacd_agent(config: DictConfig, env: gym.Env):
  'discrete observation may not work well'
  critic_base = SingleQCriticDiscrete

  if isinstance(env.observation_space, Discrete):
    obs_dim = env.observation_space.n
    discrete_obs = True
  else:
    obs_dim = env.observation_space.shape[0]
    discrete_obs = False

  if not (isinstance(env.action_space, Discrete)):
    raise RuntimeError(
        "Invalid action space: Only Discrete action spaces supported")

  action_dim = env.action_space.n
  actor = DiscreteActor(obs_dim, action_dim, config.hidden_policy)

  agent = SAC_Discrete(config, obs_dim, action_dim, discrete_obs, critic_base,
                       actor)

  return agent
