from typing import Callable, Any, List, Sequence
from collections import defaultdict
import os
import gymnasium
import torch
import pickle
import numpy as np
from gym import Env
from idil_algs.baselines.IQLearn.utils.utils import eval_mode
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


def evaluate(agent: OptionSAC, env: Env,
             num_episodes=10, vis=True, seed: int = None,
             env_name: str = ""):
  """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
      seed : seed for the environment, only to be passed if environment belongs to the gymnasium module
    Returns:
      Averaged reward and a total number of steps.
    """
  total_timesteps = []
  total_returns = []
  successes = []

  while len(total_returns) < num_episodes:
    if "franka" in env_name.lower():
      _seed = seed if seed is not None else np.random.randint(0, 1000)
      _reset_obj = env.reset(seed=_seed)
      state = _reset_obj['state']['observation']
    else:
      state = env.reset()
    prev_latent, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    done = False

    with eval_mode(agent):
      while not done:
        latent, action = agent.choose_action(state,
                                             prev_latent,
                                             prev_act,
                                             sample=False)
        if "franka" in env_name.lower():
          next_state_obj, reward, terminated, truncated, info = env.step(action)
          next_state = next_state_obj["observation"]
          done = terminated or truncated
        else:
          next_state, reward, done, info = env.step(action)
        state = next_state
        prev_latent = latent
        prev_act = action

        if "franka" in env_name.lower():
          total_returns.append(reward)
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
                     mental_states_idx: List[int]=None,
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


  if mental_states_idx is not None:
    _iterator = zip(range(len(mental_states_idx)), mental_states_idx)
  else:
    _iterator = zip(range(num_samples), range(num_samples))

  for ms_idx, traj_idx in _iterator:
    length = len(expert_traj["rewards"][traj_idx])

    dict_batch['states'].append(
        np.array(expert_traj["states"][traj_idx]).reshape(length, -1))

    dict_batch['prev_latents'].append(init_latent)
    dict_batch['prev_latents'].append(
        np.array(mental_states[ms_idx][:-1]).reshape(-1, 1))

    dict_batch['prev_actions'].append(init_action)
    dict_batch['prev_actions'].append(
        np.array(expert_traj["actions"][traj_idx][:-1]).reshape(-1, action_dim))

    dict_batch['next_states'].append(
        np.array(expert_traj["next_states"][traj_idx]).reshape(length, -1))
    dict_batch['latents'].append(np.array(mental_states[ms_idx]).reshape(-1, 1))
    dict_batch['actions'].append(
        np.array(expert_traj["actions"][traj_idx]).reshape(-1, action_dim))
    dict_batch['rewards'].append(
        np.array(expert_traj["rewards"][traj_idx]).reshape(-1, 1))
    dict_batch['dones'].append(
        np.array(expert_traj["dones"][traj_idx]).reshape(-1, 1))

    if mental_states_after_end is not None:
      dict_batch["next_latents"].append(
          np.array(mental_states[ms_idx][1:]).reshape(-1, 1))
      dict_batch["next_latents"].append(
          np.array(mental_states_after_end[ms_idx]).reshape(-1))

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


def build_expert_batch_with_topK(expert_trajectories,
                                 extra_trajectories,
                                 device,
                                 init_latent,
                                 init_action,
                                 expert_next_latents: list,
                                 extra_next_latents: list):
  '''
  given expert trajectories and extra trajectories filtered by entropy,
  build a new batch of data for training the model.
  This function appends existing expert trajectories to the filtered 
  trajectories.

  return: dictionary with these keys: states, prev_latents, prev_actions, 
                                  next_states, latents, actions, rewards, dones
  '''

  dict_batch = {}
  dict_batch['states'] = []
  dict_batch['prev_latents'] = []
  dict_batch['prev_actions'] = []
  dict_batch['next_states'] = []
  dict_batch['next_latents'] = []
  dict_batch['latents'] = []
  dict_batch['actions'] = []
  dict_batch['rewards'] = []
  dict_batch['dones'] = []

  init_latent = np.array(init_latent).reshape(-1)
  init_action = np.array(init_action).reshape(-1)
  action_dim = len(init_action)

  for idx_exp in range(len(expert_trajectories['states'])):


    length = len(expert_trajectories["rewards"][idx_exp])
    dict_batch['states'].append(np.array(expert_trajectories["states"][idx_exp]).reshape(length, -1))
    
    # include shifted previous latents
    # create latents storage from the expert trajectories
    # similarly, create actions storage from the expert trajectories
    dict_batch['prev_latents'].append(init_latent)
    dict_batch['prev_latents'].append(np.array(expert_trajectories["latents"][idx_exp][:-1]).reshape(-1, 1))
    dict_batch['prev_actions'].append(init_action)
    dict_batch['prev_actions'].append(np.array(expert_trajectories["actions"][idx_exp][:-1]).reshape(-1, action_dim))

    # include shifted next states and latents
    dict_batch['next_states'].append(np.array(expert_trajectories["next_states"][idx_exp]).reshape(length, -1))
    dict_batch['next_latents'].append(np.array(expert_next_latents[idx_exp]).reshape(-1, 1))

    dict_batch['latents'].append(np.array(expert_trajectories["latents"][idx_exp]).reshape(-1, 1))
    dict_batch['actions'].append(np.array(expert_trajectories["actions"][idx_exp]).reshape(-1, action_dim))
    dict_batch['rewards'].append(np.array(expert_trajectories["rewards"][idx_exp]).reshape(-1, 1))
    dict_batch['dones'].append(np.array(expert_trajectories["dones"][idx_exp]).reshape(-1, 1))

  for idx_extra in range(len(extra_trajectories['states'])):
    length = len(extra_trajectories["rewards"][idx_extra])
    dict_batch['states'].append(np.array(extra_trajectories["states"][idx_extra]).reshape(length, -1))

    # include shifted previous latents
    dict_batch['prev_latents'].append(init_latent)
    dict_batch['prev_latents'].append(np.array(extra_trajectories["latents"][idx_extra][:-1]).reshape(-1, 1))
    dict_batch['prev_actions'].append(init_action)
    dict_batch['prev_actions'].append(np.array(extra_trajectories["actions"][idx_extra][:-1]).reshape(-1, action_dim))

    # include shifted next states and latents
    dict_batch['next_states'].append(np.array(extra_trajectories["next_states"][idx_extra]).reshape(length, -1))
    dict_batch['next_latents'].append(np.array(extra_next_latents[idx_extra]).reshape(-1, 1))


    dict_batch['latents'].append(np.array(extra_trajectories["latents"][idx_extra]).reshape(-1, 1))
    dict_batch['actions'].append(np.array(extra_trajectories["actions"][idx_extra]).reshape(-1, action_dim))
    dict_batch['rewards'].append(np.array(extra_trajectories["rewards"][idx_extra]).reshape(-1, 1))
    dict_batch['dones'].append(np.array(extra_trajectories["dones"][idx_extra]).reshape(-1, 1))

  # assert that the total length of the dict batch is equal to the sum of 
  # the lengths of the expert and extra trajectories
  assert len(dict_batch['states']) == (len(expert_trajectories['states']) + len(extra_trajectories['states']))

  for key, val in dict_batch.items():
    tmp = np.vstack(val)
    dict_batch[key] = torch.as_tensor(tmp, dtype=torch.float, device=device)

  return dict_batch