import os
import pickle
import numpy as np
import click
from collections import defaultdict
from idil.baselines.option_gail.utils.mujoco_env import load_demo


@click.command()
@click.option("--data-path", type=str, default="", help=".torch file path")
@click.option("--save-path", type=str, default="", help="save path")
@click.option("--clip-action", type=bool, default=False, help="clip to [-1, 1]")
def conv_torch_trajs_2_iql_format(data_path, save_path, clip_action=False):
  sar_trajectories, _ = load_demo(data_path, 9999999999)  # load all
  num_traj = len(sar_trajectories)
  print("# trajectories:", num_traj)

  expert_trajs = defaultdict(list)

  for trajectory in sar_trajectories:
    s_arr, a_arr, r_arr = trajectory
    s_arr, a_arr, r_arr = s_arr.numpy(), a_arr.numpy(), r_arr.numpy()
    if clip_action:
      a_arr = np.clip(a_arr, -1, 1)

    states = s_arr[:-1]
    next_states = s_arr[1:]
    actions = a_arr[:-1]
    length = len(states)
    dones = np.zeros(length)
    rewards = r_arr[:-1]

    expert_trajs["states"].append(states.reshape(length, -1))
    expert_trajs["next_states"].append(next_states.reshape(length, -1))
    expert_trajs["actions"].append(actions.reshape(length, -1))
    expert_trajs["rewards"].append(rewards)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(length)

  with open(save_path, 'wb') as f:
    pickle.dump(expert_trajs, f)


if __name__ == "__main__":
  # experts/AntPush-v0_400_clipped.pkl
  conv_torch_trajs_2_iql_format()
