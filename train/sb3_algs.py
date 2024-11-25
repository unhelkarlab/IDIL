import gym
import random
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.util import logger as sb3_logger
from idil.IDIL.train import (load_expert_data_w_labels,
                             compute_expert_return_mean)
import wandb
import omegaconf


def sb3_bc(config, demo_path, num_trajs, log_dir, output_dir, log_interval):
  env_name = config.env_name
  seed = config.seed

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  alg_name = config.alg_name
  run_name = f"{alg_name}_{config.tag}"
  wandb.init(project=env_name,
             name=run_name,
             entity='entity-name',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  env = gym.make(env_name)  # type: gym.Env

  # Seed envs
  env.seed(seed + 10)

  # best_output_path = os.path.join(output_dir, f"best_{alg_name}_{env_name}")

  bc_logger = sb3_logger.configure(log_dir, format_strs=("tensorboard", ))

  expert_dataset, _, _ = load_expert_data_w_labels(demo_path, num_trajs, 0,
                                                   seed)

  expert_avg, expert_std = compute_expert_return_mean(
      expert_dataset.trajectories)

  wandb.run.summary["expert_avg"] = expert_avg
  wandb.run.summary["expert_std"] = expert_std

  list_trajectories = []
  for i_e in range(len(expert_dataset.trajectories["rewards"])):
    length = expert_dataset.trajectories["lengths"][i_e]
    states = np.array(expert_dataset.trajectories["states"][i_e])
    last_state = np.array(expert_dataset.trajectories["next_states"][i_e][-1])
    states = np.concatenate(
        [states.reshape(length, -1),
         last_state.reshape(1, -1)], axis=0)
    done = expert_dataset.trajectories["dones"][i_e][-1]

    actions = expert_dataset.trajectories["actions"][i_e]
    list_trajectories.append(
        Trajectory(obs=states,
                   acts=np.array(actions).reshape(length, -1),
                   terminal=done,
                   infos=None))

  transitions = rollout.flatten_trajectories(list_trajectories)

  dummy_scheduler = lambda _: torch.finfo(torch.float32).max

  policy = ActorCriticPolicy(observation_space=env.observation_space,
                             action_space=env.action_space,
                             lr_schedule=dummy_scheduler,
                             net_arch=dict(pi=config.hidden_policy,
                                           vf=config.hidden_critic),
                             activation_fn=nn.ReLU)

  bc_trainer = BC(observation_space=env.observation_space,
                  action_space=env.action_space,
                  policy=policy,
                  rng=np.random.default_rng(seed),
                  demonstrations=transitions,
                  custom_logger=bc_logger,
                  batch_size=config.mini_batch_size,
                  optimizer_cls=torch.optim.Adam,
                  optimizer_kwargs={'lr': config.optimizer_lr_policy})

  bc_trainer.train(n_batches=config.n_batches, log_interval=log_interval)

  epi_rewards = []
  successes = []
  for _ in range(8):
    state = env.reset()
    epi_reward = 0
    done = False
    while not done:
      with torch.no_grad():
        action, _ = policy.predict(state)
        state, reward, done, info = env.step(action)
        epi_reward += reward
    epi_rewards.append(epi_reward)

    if 'task_success' in info.keys():
      successes.append(info['task_success'])

  best_returns = np.mean(epi_rewards)
  wandb.run.summary["best_returns"] = best_returns
  wandb.finish()
  print(f"Best returns: {best_returns}")
  if len(successes) > 0:
    success_rate = np.mean(successes)
    print(f"Success rate: {success_rate}")
