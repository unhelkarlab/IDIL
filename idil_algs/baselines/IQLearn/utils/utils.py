import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Callable, Any, Sequence
import pickle
import gym
import gymnasium
from collections import defaultdict
from stable_baselines3.common.monitor import Monitor
from .normalize_action_wrapper import (check_and_normalize_box_actions)


def conv_trajectories_2_iql_format(sa_trajectories: Sequence,
                                   cb_conv_action_to_idx: Callable[[Any], int],
                                   cb_get_reward: Callable[[Any, Any],
                                                           float], path: str):
  'sa_trajectories: okay to include the terminal state'
  expert_trajs = defaultdict(list)

  for trajectory in sa_trajectories:
    traj = []
    for t in range(len(trajectory) - 1):
      cur_tup = trajectory[t]
      next_tup = trajectory[t + 1]

      state, action = cur_tup[0], cur_tup[1]
      next_state, next_action = next_tup[0], next_tup[1]

      aidx = cb_conv_action_to_idx(action)
      reward = cb_get_reward(state, action)

      done = next_action is None
      traj.append((state, next_state, aidx, reward, done))

    states, next_states, actions, rewards, dones = zip(*traj)

    expert_trajs["states"].append(states)
    expert_trajs["next_states"].append(next_states)
    expert_trajs["actions"].append(actions)
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(len(traj))

  print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
  with open(path, 'wb') as f:
    pickle.dump(expert_trajs, f)


def make_env(env_name, monitor=True, env_make_kwargs={}):
  env_make_kwargs = env_make_kwargs or {}
  if "franka" in env_name.lower():
    env = gymnasium.make(env_name, **env_make_kwargs)
  else:
    env = gym.make(env_name, **env_make_kwargs)  
    if monitor:
      env = Monitor(env, "gym")

  # Normalize box actions to [-1, 1]
  env = check_and_normalize_box_actions(env)
  return env


def one_hot(indices: torch.Tensor, num_classes):
  return F.one_hot(indices.reshape(-1).long(),
                   num_classes=num_classes).to(dtype=torch.float)


def one_hot_w_nan(indices: torch.Tensor, num_classes):
  indices_flat = indices.reshape(-1)

  len_ind = len(indices_flat)
  one_hot_tensor = torch.zeros((len_ind, num_classes),
                               dtype=torch.float).to(device=indices.device)

  mask_non_nan = ~indices_flat.isnan()
  valid_indices = indices_flat[mask_non_nan]
  if len(valid_indices) != 0:
    one_hot_tensor[mask_non_nan] = F.one_hot(
        valid_indices.long(), num_classes=num_classes).to(dtype=torch.float)

  return one_hot_tensor


class eval_mode(object):

  def __init__(self, *models):
    self.models = models

  def __enter__(self):
    self.prev_states = []
    for model in self.models:
      self.prev_states.append(model.training)
      model.train(False)

  def __exit__(self, *args):
    for model, state in zip(self.models, self.prev_states):
      model.train(state)
    return False


def evaluate(actor, env, num_episodes=10, vis=True):
  """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
  total_timesteps = []
  total_returns = []
  successes = []

  while len(total_returns) < num_episodes:
    state = env.reset()
    done = False

    with eval_mode(actor):
      while not done:
        action = actor.choose_action(state, sample=False)
        next_state, reward, done, info = env.step(action)
        state = next_state

        if 'episode' in info.keys():
          total_returns.append(info['episode']['r'])
          total_timesteps.append(info['episode']['l'])

    if 'task_success' in info.keys():
      successes.append(info['task_success'])

  return total_returns, total_timesteps, successes


def weighted_softmax(x, weights):
  x = x - torch.max(x, dim=0)[0]
  return weights * torch.exp(x) / torch.sum(
      weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
  for param, target_param in zip(net.parameters(), target_net.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update(source, target):
  for param, target_param in zip(source.parameters(), target.parameters()):
    target_param.data.copy_(param.data)


def weight_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)


def mlp(input_dim, output_dim, list_hidden_dims, output_mod=None):
  if len(list_hidden_dims) == 0:
    mods = [nn.Linear(input_dim, output_dim)]
  else:
    mods = [nn.Linear(input_dim, list_hidden_dims[0]), nn.ReLU(inplace=True)]
    for i in range(len(list_hidden_dims) - 1):
      mods += [
          nn.Linear(list_hidden_dims[i], list_hidden_dims[i + 1]),
          nn.ReLU(inplace=True)
      ]
    mods.append(nn.Linear(list_hidden_dims[-1], output_dim))
  if output_mod is not None:
    mods.append(output_mod)
  trunk = nn.Sequential(*mods)
  return trunk


def get_concat_samples(policy_batch, expert_batch, is_sqil: bool = False):
  '''
  policy_batch, expert_batch: the 2nd last item should be reward,
                                and the last item should be done
  return: concatenated batch with an additional item of is_expert
  '''
  concat_batch = []

  reward_idx = len(policy_batch) - 2
  for idx in range(reward_idx):
    concat_batch.append(torch.cat([policy_batch[idx], expert_batch[idx]],
                                  dim=0))

  # ----- concat reward data
  online_batch_reward = policy_batch[reward_idx]
  expert_batch_reward = expert_batch[reward_idx]
  if is_sqil:
    # convert policy reward to 0
    online_batch_reward = torch.zeros_like(online_batch_reward)
    # convert expert reward to 1
    expert_batch_reward = torch.ones_like(expert_batch_reward)
  concat_batch.append(
      torch.cat([online_batch_reward, expert_batch_reward], dim=0))

  # ----- concat done data
  concat_batch.append(torch.cat([policy_batch[-1], expert_batch[-1]], dim=0))

  # ----- mark what is expert data and what is online data
  is_expert = torch.cat([
      torch.zeros_like(online_batch_reward, dtype=torch.bool),
      torch.ones_like(expert_batch_reward, dtype=torch.bool)
  ],
                        dim=0)
  concat_batch.append(is_expert)

  return concat_batch


def average_dicts(dict1, dict2):
  return {
      key: 1 / 2 * (dict1.get(key, 0) + dict2.get(key, 0))
      for key in set(dict1) | set(dict2)
  }


def compute_expert_return_mean(trajectories):
  expert_returns = []
  n_expert_trj = len(trajectories["rewards"])
  for i_e in range(n_expert_trj):
    expert_returns.append(sum(trajectories["rewards"][i_e]))

  expert_return_avg = np.mean(expert_returns)
  expert_return_std = np.std(expert_returns)
  print(f'Demo reward: {expert_return_avg} +- {expert_return_std}')
  return expert_return_avg, expert_return_std