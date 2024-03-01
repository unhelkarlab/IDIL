import torch
from ..model.option_policy import OptionPolicy, Policy
from ..model.option_policy_v2 import OptionPolicyV2
import numpy as np
from typing import Union
import os
import random
from ml_algs.baselines.IQLearn.utils.utils import compute_expert_return_mean
from ml_algs.baselines.IQLearn.dataset.expert_dataset import ExpertDataset


def sample_batch(policy: Union[OptionPolicy, Policy], agent, n_step):
  sample = agent.collect(policy.state_dict(), n_step, fixed=False)
  rsum = sum([sxar[-1].sum().item() for sxar in sample]) / len(sample)
  avgsteps = sum([sxar[-1].size(0) for sxar in sample]) / len(sample)
  return sample, rsum, avgsteps


def validate(policy: Union[OptionPolicy, Policy], sa_array):
  with torch.no_grad():
    log_pi = 0.
    cs = []
    for s_array, a_array in sa_array:
      if isinstance(policy, OptionPolicy) or isinstance(policy, OptionPolicyV2):
        c_array, logp = policy.viterbi_path(s_array, a_array)
        log_pi += logp.item()
        cs.append(c_array.detach().cpu().squeeze(dim=-1).numpy())
      else:
        log_pi += policy.log_prob_action(s_array, a_array).sum().item()
        cs.append([0.])
    log_pi /= len(sa_array)
  return log_pi, cs


def reward_validate(agent,
                    policy: Union[OptionPolicy, Policy],
                    n_sample=-8,
                    do_print=True):
  trajs = agent.collect(policy.state_dict(), n_sample, fixed=True)
  rsums = [tr[-1].sum().item() for tr in trajs]
  steps = [tr[-1].size(0) for tr in trajs]
  successes = [tr[-2] for tr in trajs]
  if isinstance(policy, OptionPolicy) or isinstance(policy, OptionPolicyV2):
    css = [
        tr[1].cpu().squeeze(dim=-1).numpy()
        for _, tr in sorted(zip(rsums, trajs), key=lambda d: d[0], reverse=True)
    ]
  else:
    css = None

  info_dict = {
      "episode_reward": np.mean(rsums),
      "episode_step": np.mean(steps),
  }

  if successes[0] is not None:
    info_dict["success_rate"] = np.mean(successes)

  if do_print:
    print(f"R: [ {np.min(rsums):.02f} ~ {np.max(rsums):.02f},",
          f"avg: {info_dict['episode_reward']:.02f} ],",
          f"L: [ {np.min(steps)} ~ {np.max(steps)}, ",
          f"avg: {info_dict['episode_step']:.02f} ]")
  return info_dict, css


def lr_factor_func(i_iter, end_iter, start=1., end=0.):
  if i_iter <= end_iter:
    return start - (start - end) * i_iter / end_iter
  else:
    return end


def env_class(env_type):
  if env_type == "rlbench":
    raise NotImplementedError("RLBench is not supported.")
  else:  # env_type == "mujoco":
    from .mujoco_env import MujocoEnv as RLEnv

  return RLEnv


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)


def load_n_convert_data(demo_path, n_traj, n_labeled, device, dim_c, seed):
  expert_dataset = ExpertDataset(demo_path, n_traj, 1, seed + 42)
  trajectories = expert_dataset.trajectories
  expert_avg, expert_std = compute_expert_return_mean(trajectories)

  cnt_label = 0
  demo_labels = []
  demo_sa_array = []
  for epi in range(n_traj):
    n_steps = len(trajectories["rewards"][epi])
    s_array = torch.as_tensor(trajectories["states"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    a_array = torch.as_tensor(trajectories["actions"][epi],
                              dtype=torch.float32).reshape(n_steps, -1)
    if "latents" in trajectories:
      x_array = torch.zeros(n_steps + 1, 1, dtype=torch.long, device=device)
      x_array[1:] = torch.as_tensor(trajectories["latents"][epi],
                                    dtype=torch.float32).reshape(n_steps, -1)
      x_array[0] = dim_c
    else:
      x_array = None

    demo_sa_array.append((s_array.to(device), a_array.to(device)))
    if epi < n_labeled:
      demo_labels.append(x_array.to(device))
      cnt_label += 1
    else:
      demo_labels.append(None)

  demo_sa_array = tuple(demo_sa_array)

  print(f"num_labeled: {cnt_label} / {n_traj}, num_samples: ",
        len(expert_dataset))

  return demo_sa_array, demo_labels, cnt_label, expert_avg, expert_std
