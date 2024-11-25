from typing import Optional
import os
import torch

import gym
from ..model.option_policy import Policy
from .state_filter import StateFilter
from gym.spaces import Discrete, Box
from omegaconf import DictConfig


class MujocoEnv(object):

  def __init__(self, task_name: str = "HalfCheetah-v2"):
    self.task_name = task_name
    self.env = None
    self.display = False

  def init(self, display=False):
    self.env = gym.make(self.task_name)
    self.display = display
    return self

  def seed(self, seed):
    self.env.seed(seed)

  def reset(self, random: bool = False):
    s = self.env.reset()
    return s

  def step(self, a):
    s, reward, terminate, info = self.env.step(a)
    if self.display:
      self.env.render()
    return s, reward, terminate, info

  def state_action_size(self):
    if self.env is not None:
      if isinstance(self.env.observation_space, Discrete):
        s_dim = self.env.observation_space.n
      else:
        s_dim = self.env.observation_space.shape[0]

      if isinstance(self.env.action_space, Discrete):
        a_dim = self.env.action_space.n
      else:
        a_dim = self.env.action_space.shape[0]
    else:
      env = gym.make(self.task_name)
      if isinstance(env.observation_space, Discrete):
        s_dim = env.observation_space.n
      else:
        s_dim = env.observation_space.shape[0]

      if isinstance(env.action_space, Discrete):
        a_dim = env.action_space.n
      else:
        a_dim = env.action_space.shape[0]
      env.close()
    return s_dim, a_dim

  def is_discrete_state_action(self):
    if self.env is not None:
      if isinstance(self.env.observation_space, Discrete):
        discrete_s = True
      else:
        discrete_s = False

      if isinstance(self.env.action_space, Discrete):
        discrete_a = True
      else:
        discrete_a = False
    else:
      env = gym.make(self.task_name)
      if isinstance(env.observation_space, Discrete):
        discrete_s = True
      else:
        discrete_s = False

      if isinstance(env.action_space, Discrete):
        discrete_a = True
      else:
        discrete_a = False

      env.close()
    return discrete_s, discrete_a


def load_demo(load_path: str, n_traj: int = 10):
  if not os.path.isfile(load_path):
    return None, None

  print(f"Demo Loaded from {load_path}")
  samples, filter_state = torch.load(load_path)
  n_current_demo = 0
  sample = []
  for idx, traj in enumerate(samples):
    if idx >= n_traj:
      break
    sample.append(traj)
    n_current_demo += traj[2].size(0)
  print(
      f"Loaded {len(sample)} episodes with a total of {n_current_demo} samples."
  )

  return sample, filter_state


def generate_demo(mujoco_config: DictConfig,
                  save_path: str,
                  expert_path: Optional[str],
                  config_path: Optional[str] = None,
                  n_traj: int = 10,
                  display: bool = False):

  if config_path is not None and os.path.isfile(config_path):
    mujoco_config.load_saved(config_path)

  mujoco_config.device = "cpu"
  use_rs = mujoco_config.use_state_filter

  env = MujocoEnv(mujoco_config.env_name)
  dim_s, dim_a = env.state_action_size()
  env.init(display=display)

  policy_state, filter_state = torch.load(expert_path, map_location="cpu")
  policy = Policy(mujoco_config, dim_s, dim_a)
  policy.load_state_dict(policy_state)
  rs = StateFilter(enable=use_rs)
  rs.load_state_dict(filter_state)

  sample = []
  cnt = 0
  n_current_demo = 0
  while cnt < n_traj:
    with torch.no_grad():
      s_array = []
      a_array = []
      r_array = []
      s, done = env.reset(), False
      while not done:
        st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(dim=0)
        s_array.append(st.clone())
        at = policy.sample_action(rs(st, fixed=True), fixed=True)
        a_array.append(at.clone())
        s, r, done, info = env.step(at.squeeze(dim=0).numpy())
        r_array.append(r)
      a_array = torch.cat(a_array, dim=0)
      s_array = torch.cat(s_array, dim=0)
      r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=1)
      print(f"R-Sum={r_array.sum()}, L={r_array.size(0)}")
      keep = input(f"{cnt}/{n_traj} Keep this ? [y|n]>>>")
      if keep == 'y':
        sample.append((s_array, a_array, r_array))
        n_current_demo += r_array.size(0)
        cnt += 1
  torch.save((sample, rs.state_dict()), save_path)

  print(
      f"Generated {n_traj} episodes with a total of {n_current_demo} samples.")
  return sample, filter_state


def get_demo(mujoco_config: DictConfig,
             path: str,
             expert_path: Optional[str] = None,
             config_path: Optional[str] = None,
             n_traj: int = 10,
             display: bool = False):
  sample, filter_state = load_demo(path, n_traj)
  if sample is None:
    if expert_path is None:
      raise ValueError("expert path is missing")

    sample, filter_state = generate_demo(mujoco_config, path, expert_path,
                                         config_path, n_traj, display)

  return sample, filter_state


if __name__ == "__main__":
  generate_demo(display=True)
