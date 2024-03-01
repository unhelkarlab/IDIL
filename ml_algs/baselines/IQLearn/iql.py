from typing import Optional
import os
import random
import numpy as np
import time
import datetime
import torch
from collections import deque
from .utils.utils import (make_env, eval_mode, average_dicts,
                          get_concat_samples, evaluate, soft_update,
                          hard_update, compute_expert_return_mean)

from .agent import make_sac_agent, make_softq_agent, make_sacd_agent
from .agent.softq_models import SimpleQNetwork, SingleQCriticDiscrete
from .agent.sac_models import DoubleQCritic, SingleQCritic
from .dataset.memory import Memory
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from .utils.logger import Logger
from itertools import count
import types
from .iq import iq_loss, OFFLINE_METHOD_LOSS
import wandb
import omegaconf


def run_sac(config: omegaconf.DictConfig,
            log_dir,
            output_dir,
            log_interval=500,
            eval_interval=2000,
            env_kwargs={}):
  return trainer_impl(config,
                      None,
                      None,
                      log_dir,
                      output_dir,
                      log_interval,
                      eval_interval,
                      env_kwargs,
                      imitate=False)


def run_iql(config: omegaconf.DictConfig,
            demo_path,
            num_trajs,
            log_dir,
            output_dir,
            log_interval=500,
            eval_interval=2000,
            env_kwargs={}):
  return trainer_impl(config,
                      demo_path,
                      num_trajs,
                      log_dir,
                      output_dir,
                      log_interval,
                      eval_interval,
                      env_kwargs,
                      imitate=True)


def trainer_impl(config: omegaconf.DictConfig,
                 demo_path,
                 num_trajs,
                 log_dir,
                 output_dir,
                 log_interval=500,
                 eval_interval=2000,
                 env_kwargs={},
                 imitate=True):
  'agent_name: softq / sac / sacd'
  env_name = config.env_name
  seed = config.seed
  batch_size = config.mini_batch_size
  replay_mem = config.n_sample
  eps_window = 10
  num_explore_steps = config.max_explore_step
  agent_name = config.iql_agent_name
  output_suffix = ""
  load_path = None

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  run_name = f"{config.alg_name}_{config.tag}"
  wandb.init(project=env_name,
             name=run_name,
             entity='sangwon-seo',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # constants
  num_episodes = 8
  save_interval = 10
  only_observabtion_based = False

  # device
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  cuda_deterministic = False

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  device = torch.device(device_name)
  if device.type == 'cuda' and torch.cuda.is_available() and cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  env = make_env(env_name, env_make_kwargs=env_kwargs)
  eval_env = make_env(env_name, env_make_kwargs=env_kwargs)

  # Seed envs
  env.seed(seed)
  eval_env.seed(seed + 10)

  if agent_name == "softq":
    use_target = False
    do_soft_update = False
    agent = make_softq_agent(config, env)
  elif agent_name == "sac":
    use_target = True
    do_soft_update = True
    agent = make_sac_agent(config, env)
  elif agent_name == "sacd":
    use_target = True
    do_soft_update = True
    agent = make_sacd_agent(config, env)
  else:
    raise NotImplementedError

  if load_path is not None:
    if os.path.isfile(load_path):
      print("=> loading pretrain '{}'".format(load_path))
      agent.load(load_path)
    else:
      print("[Attention]: Did not find checkpoint {}".format(load_path))

  initial_mem = config.init_sample
  replay_mem = int(replay_mem)
  initial_mem = int(initial_mem)
  assert initial_mem <= replay_mem

  # Load expert data
  if imitate:
    subsample_freq = 1
    expert_memory_replay = Memory(replay_mem, seed)
    expertdata = expert_memory_replay.load(demo_path,
                                           num_trajs=num_trajs,
                                           sample_freq=subsample_freq,
                                           seed=seed + 42)
    batch_size = min(batch_size, expert_memory_replay.size())
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    expert_return_avg, expert_return_std = compute_expert_return_mean(
        expertdata.trajectories)
    wandb.run.summary["expert_avg"] = expert_return_avg
    wandb.run.summary["expert_std"] = expert_return_std

  online_memory_replay = Memory(replay_mem, seed + 1)

  eps_window = int(eps_window)
  num_explore_steps = int(num_explore_steps)

  # Setup logging
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  agent=agent_name,
                  run_name=f"{env_name}_{run_name}")

  # track mean reward and scores
  rewards_window = deque(maxlen=eps_window)  # last N rewards
  epi_step_window = deque(maxlen=eps_window)
  success_window = deque(maxlen=eps_window)
  best_eval_returns = -np.inf
  best_success_rate = -np.inf
  cnt_steps = 0

  begin_learn = False
  episode_reward = 0
  explore_steps = 0
  update_count = 0

  for epoch in count():
    state = env.reset()
    episode_reward = 0
    done = False

    for episode_step in count():

      with eval_mode(agent):
        # if not begin_learn:
        #   action = env.action_space.sample()
        # else:
        action = agent.choose_action(state, sample=True)
      next_state, reward, done, info = env.step(action)
      episode_reward += reward

      if explore_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps, successes = evaluate(
            agent, eval_env, num_episodes=num_episodes)
        returns = np.mean(eval_returns)
        # explore_steps += 1  # To prevent repeated eval at timestep 0
        logger.log('eval/episode', epoch, explore_steps)
        logger.log('eval/episode_reward', returns, explore_steps)
        logger.log('eval/episode_step', np.mean(eval_timesteps), explore_steps)
        if len(successes) > 0:
          success_rate = np.mean(successes)
          logger.log('eval/success_rate', success_rate, explore_steps)
          if success_rate > best_success_rate:
            best_success_rate = success_rate
            wandb.run.summary["best_success_rate"] = best_success_rate

        logger.dump(explore_steps, ty='eval')

        if returns > best_eval_returns:
          # Store best eval returns
          best_eval_returns = returns
          wandb.run.summary["best_returns"] = best_eval_returns
          save(agent,
               epoch,
               1,
               env_name,
               imitate,
               output_dir=output_dir,
               suffix=output_suffix + "_best")

      # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((state, next_state, action, reward, done_no_lim))

      explore_steps += 1
      if online_memory_replay.size() >= initial_mem:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        if explore_steps >= num_explore_steps:
          print('Finished!')
          wandb.finish()
          return

        ######
        losses = {}
        if explore_steps % config.update_interval == 0:
          if imitate:
            # IQ-Learn Modification
            agent.iq_update = types.MethodType(iq_update, agent)
            agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
            losses = agent.iq_update(online_memory_replay, expert_memory_replay,
                                     logger, update_count,
                                     only_observabtion_based, use_target,
                                     do_soft_update, config.method_loss,
                                     config.method_regularize,
                                     config.method_div)
            update_count += 1
          else:
            losses = agent.update(online_memory_replay, logger, explore_steps)

        if explore_steps % log_interval == 0:
          for key, loss in losses.items():
            writer.add_scalar("loss/" + key, loss, global_step=explore_steps)

      if done:
        break
      state = next_state

    rewards_window.append(episode_reward)
    epi_step_window.append(episode_step + 1)
    if 'task_success' in info.keys():
      success_window.append(info['task_success'])

    cnt_steps += episode_step + 1
    if cnt_steps >= log_interval:
      cnt_steps = 0
      logger.log('train/episode', epoch, explore_steps)
      logger.log('train/episode_reward', np.mean(rewards_window), explore_steps)
      logger.log('train/episode_step', np.mean(epi_step_window), explore_steps)
      if len(success_window) > 0:
        logger.log('train/success_rate', np.mean(success_window), explore_steps)

      logger.dump(explore_steps, save=begin_learn)

    # save(agent,
    #      epoch,
    #      save_interval,
    #      env_name,
    #      agent_name,
    #      imitate,
    #      output_dir=output_dir,
    #      suffix=output_suffix)


def save(agent,
         epoch,
         save_interval,
         env_name,
         imitate: bool,
         output_dir='results',
         suffix=""):
  if epoch % save_interval == 0:
    if imitate:
      name = f'iq_{env_name}'
    else:
      name = f'rl_{env_name}'

    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    file_path = os.path.join(output_dir, f'{name}' + suffix)
    agent.save(file_path)


def iq_update_critic(self,
                     policy_batch,
                     expert_batch,
                     logger,
                     step,
                     only_observabtion_based=False,
                     use_target=False,
                     method_loss="value",
                     method_regularize=True,
                     method_div=""):
  (policy_obs, policy_next_obs, policy_action, policy_reward,
   policy_done) = policy_batch
  (expert_obs, expert_next_obs, expert_action, expert_reward,
   expert_done) = expert_batch

  if only_observabtion_based:
    # Use policy actions instead of experts actions for IL with only observations
    expert_batch = (expert_obs, expert_next_obs, policy_action, expert_reward,
                    expert_done)

  obs, next_obs, action, _, done, is_expert = get_concat_samples(
      policy_batch, expert_batch, False)
  vec_v_args = (obs, )
  vec_next_v_args = (next_obs, )
  vec_actions = (action, )

  agent = self

  current_Q = self.critic(obs, action, both=True)
  if isinstance(current_Q, tuple):
    q1_loss, loss_dict1 = iq_loss(agent, current_Q[0], vec_v_args,
                                  vec_next_v_args, vec_actions, done, is_expert,
                                  use_target, method_loss, method_regularize,
                                  method_div)
    q2_loss, loss_dict2 = iq_loss(agent, current_Q[1], vec_v_args,
                                  vec_next_v_args, vec_actions, done, is_expert,
                                  use_target, method_loss, method_regularize,
                                  method_div)
    critic_loss = 1 / 2 * (q1_loss + q2_loss)
    # merge loss dicts
    loss_dict = average_dicts(loss_dict1, loss_dict2)
  else:
    critic_loss, loss_dict = iq_loss(agent, current_Q, vec_v_args,
                                     vec_next_v_args, vec_actions, done,
                                     is_expert, use_target, method_loss,
                                     method_regularize, method_div)

  # logger.log('train/critic_loss', critic_loss, step)

  # Optimize the critic
  self.critic_optimizer.zero_grad()
  critic_loss.backward()
  if hasattr(self, 'clip_grad_val') and self.clip_grad_val:
    nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
  # step critic
  self.critic_optimizer.step()
  return loss_dict


def iq_update(self,
              policy_buffer,
              expert_buffer,
              logger,
              update_count,
              only_observabtion_based=False,
              use_target=False,
              do_soft_update=False,
              method_loss="value",
              method_regularize=True,
              method_div=""):
  policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
  expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

  losses = self.iq_update_critic(policy_batch, expert_batch, logger,
                                 update_count, only_observabtion_based,
                                 use_target, method_loss, method_regularize,
                                 method_div)

  # args
  vdice_actor = False
  offline = False
  num_actor_updates = 1

  if self.actor and update_count % self.actor_update_frequency == 0:
    if not vdice_actor:

      if offline:
        obs = expert_batch[0]
      else:
        # Use both policy and expert observations
        obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

      if num_actor_updates:
        for i in range(num_actor_updates):
          actor_alpha_losses = self.update_actor_and_alpha(
              obs, logger, update_count)

      losses.update(actor_alpha_losses)

  if use_target and update_count % self.critic_target_update_frequency == 0:
    if do_soft_update:
      soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
    else:
      hard_update(self.critic_net, self.critic_target_net)

  return losses
