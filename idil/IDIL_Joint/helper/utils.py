from typing import Callable, Any, Sequence
from collections import defaultdict
import os
import torch
import pickle
import numpy as np
from gym import Env
from idil.baselines.IQLearn.utils.utils import eval_mode
from ..agent.option_sac import OptionSAC


def conv_trajectories_2_iql_format(sax_trajectories: Sequence,
                                   cb_conv_action_to_idx: Callable[[Any], int],
                                   cb_get_reward: Callable[[Any, Any],
                                                           float], path: str):
  'sa_trajectories: okay to include the terminal state'
  expert_trajs = defaultdict(list)

  for trajectory in sax_trajectories:
    traj = []
    for t in range(len(trajectory) - 1):
      cur_tup = trajectory[t]
      next_tup = trajectory[t + 1]

      state, action, latent = cur_tup[0], cur_tup[1], cur_tup[2]
      nxt_state, nxt_action, _ = next_tup[0], next_tup[1], next_tup[2]

      aidx = cb_conv_action_to_idx(action)
      reward = cb_get_reward(state, action)

      done = nxt_action is None
      traj.append((state, nxt_state, aidx, latent, reward, done))

    len_traj = len(traj)

    unzipped_traj = zip(*traj)
    states, next_states, actions, latents, rewards, dones = map(
        np.array, unzipped_traj)

    expert_trajs["states"].append(states.reshape(len_traj, -1))
    expert_trajs["next_states"].append(next_states.reshape(len_traj, -1))
    expert_trajs["actions"].append(actions.reshape(len_traj, -1))
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(len(traj))
    expert_trajs["latents"].append(latents.reshape(len_traj, -1))

  print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
  with open(path, 'wb') as f:
    pickle.dump(expert_trajs, f)


def save(agent: OptionSAC,
         epoch,
         save_interval,
         env_name,
         alg_type: str,
         output_dir='results',
         suffix=""):
  if epoch % save_interval == 0:
    name = f'{alg_type}_{env_name}'

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    file_path = os.path.join(output_dir, f'{name}' + suffix)
    agent.save(file_path)


def evaluate(agent: OptionSAC, env: Env, num_episodes=10, vis=True):
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
    prev_latent, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    done = False

    with eval_mode(agent):
      while not done:
        latent, action = agent.choose_action(state,
                                             prev_latent,
                                             prev_act,
                                             sample=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        prev_latent = latent
        prev_act = action

        if 'episode' in info.keys():
          total_returns.append(info['episode']['r'])
          total_timesteps.append(info['episode']['l'])

    if 'task_success' in info.keys():
      successes.append(info['task_success'])

  return total_returns, total_timesteps, successes


def get_expert_batch(expert_traj,
                     mental_states,
                     device,
                     init_latent,
                     init_action,
                     mental_states_after_end=None):
  '''
  return: dictionary with these keys: states, prev_latents, prev_actions, 
                                  next_states, latents, actions, rewards, dones
  '''
  num_samples = len(expert_traj["states"])

  dict_batch = {}
  dict_batch['states'] = []
  dict_batch['prev_latents'] = []
  dict_batch['prev_actions'] = []
  dict_batch['next_states'] = []
  dict_batch['latents'] = []
  dict_batch['actions'] = []
  dict_batch['rewards'] = []
  dict_batch['dones'] = []

  init_latent = np.array(init_latent).reshape(-1)
  init_action = np.array(init_action).reshape(-1)
  action_dim = len(init_action)

  if mental_states_after_end is not None:
    dict_batch['next_latents'] = []

  for i_e in range(num_samples):
    length = len(expert_traj["rewards"][i_e])

    dict_batch['states'].append(
        np.array(expert_traj["states"][i_e]).reshape(length, -1))

    dict_batch['prev_latents'].append(init_latent)
    dict_batch['prev_latents'].append(
        np.array(mental_states[i_e][:-1]).reshape(-1, 1))

    dict_batch['prev_actions'].append(init_action)
    dict_batch['prev_actions'].append(
        np.array(expert_traj["actions"][i_e][:-1]).reshape(-1, action_dim))

    dict_batch['next_states'].append(
        np.array(expert_traj["next_states"][i_e]).reshape(length, -1))
    dict_batch['latents'].append(np.array(mental_states[i_e]).reshape(-1, 1))
    dict_batch['actions'].append(
        np.array(expert_traj["actions"][i_e]).reshape(-1, action_dim))
    dict_batch['rewards'].append(
        np.array(expert_traj["rewards"][i_e]).reshape(-1, 1))
    dict_batch['dones'].append(
        np.array(expert_traj["dones"][i_e]).reshape(-1, 1))

    if mental_states_after_end is not None:
      dict_batch["next_latents"].append(
          np.array(mental_states[i_e][1:]).reshape(-1, 1))
      dict_batch["next_latents"].append(
          np.array(mental_states_after_end[i_e]).reshape(-1))

  for key, val in dict_batch.items():
    tmp = np.vstack(val)
    dict_batch[key] = torch.as_tensor(tmp, dtype=torch.float, device=device)

  return dict_batch


def get_samples(batch_size, dataset):
  indexes = np.random.choice(np.arange(len(dataset[0])),
                             size=batch_size,
                             replace=False)

  batch = []
  for col in range(len(dataset)):
    batch.append(dataset[col][indexes])

  return batch
