import os
from tqdm import tqdm
import random
import numpy as np
import torch
from .utils.utils import (make_env, average_dicts, evaluate, soft_update,
                          hard_update, compute_expert_return_mean)

from .agent import make_sac_agent, make_softq_agent, make_sacd_agent
from .dataset.memory import Memory
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from .utils.logger import Logger
import types
from .iq import iq_loss, OFFLINE_METHOD_LOSS
import wandb
import omegaconf


def run_iql(config: omegaconf.DictConfig,
            demo_path,
            num_trajs,
            log_dir,
            output_dir,
            log_interval=500,
            eval_interval=2000,
            env_kwargs={}):
  'agent_name: softq / sac / sacd'
  env_name = config.env_name
  seed = config.seed
  batch_size = config.mini_batch_size
  agent_name = config.iql_agent_name
  output_suffix = ""
  max_step = int(config.max_explore_step)

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  run_name = f"{config.alg_name}_{config.tag}"
  wandb.init(project=env_name + '_offline',
             name=run_name,
             entity='sangwon-seo',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # constants
  num_episodes = 8

  # device
  device_name = config.device
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

  # Seed envs
  env.seed(seed + 10)

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

  # Load expert data
  expert_memory_replay = Memory(int(config.n_sample), seed)
  expertdata = expert_memory_replay.load(demo_path,
                                         num_trajs=num_trajs,
                                         sample_freq=1,
                                         seed=seed + 42)

  batch_size = min(batch_size, expert_memory_replay.size())
  print(f'--> Expert memory size: {expert_memory_replay.size()}')

  expert_return_avg, expert_return_std = compute_expert_return_mean(
      expertdata.trajectories)
  wandb.run.summary["expert_avg"] = expert_return_avg
  wandb.run.summary["expert_std"] = expert_return_std

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
  best_eval_returns = -np.inf
  best_success_rate = -np.inf

  for update_steps in tqdm(range(1, max_step + 1)):
    expert_batch = expert_memory_replay.get_samples(batch_size, device)

    agent.iq_offline = types.MethodType(iq_offline, agent)
    losses = agent.iq_offline(expert_batch, logger, update_steps, use_target,
                              do_soft_update, config.method_regularize,
                              config.method_div)
    if update_steps % log_interval == 0:
      for key, loss in losses.items():
        writer.add_scalar("loss/" + key, loss, global_step=update_steps)

    if update_steps % eval_interval == 0 or update_steps == max_step:
      eval_returns, eval_timesteps, successes = evaluate(
          agent, env, num_episodes=num_episodes)
      returns = np.mean(eval_returns)
      # update_steps += 1  # To prevent repeated eval at timestep 0
      logger.log('eval/episode_reward', returns, update_steps)
      logger.log('eval/episode_step', np.mean(eval_timesteps), update_steps)
      if len(successes) > 0:
        success_rate = np.mean(successes)
        logger.log('eval/success_rate', success_rate, update_steps)
        print("succss rate", success_rate)
        if success_rate > best_success_rate:
          best_success_rate = success_rate
          wandb.run.summary["best_success_rate"] = best_success_rate

      logger.dump(update_steps, ty='eval')

      if returns > best_eval_returns:
        # Store best eval returns
        best_eval_returns = returns
        wandb.run.summary["best_returns"] = best_eval_returns
        save(agent,
             update_steps,
             1,
             env_name,
             True,
             output_dir=output_dir,
             suffix=output_suffix + "_best")

  print('Finished!')
  wandb.finish()
  return


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


def iq_offline(self,
               expert_batch,
               logger,
               update_count,
               use_target=False,
               do_soft_update=False,
               method_regularize=True,
               method_div="chi"):
  (expert_obs, expert_next_obs, expert_action, expert_reward,
   expert_done) = expert_batch

  is_expert = torch.ones_like(expert_reward, dtype=torch.bool)

  vec_v_args = (expert_obs, )
  vec_next_v_args = (expert_next_obs, )
  vec_actions = (expert_action, )

  agent = self

  # for offline setting these shouldn't be changed
  if method_regularize or method_div == "chi":
    # apply only one (same effect)
    method_regularize = False
    method_div = "chi"

  current_Q = self.critic(expert_obs, expert_action, both=True)
  if isinstance(current_Q, tuple):
    q1_loss, loss_dict1 = iq_loss(agent, current_Q[0], vec_v_args,
                                  vec_next_v_args, vec_actions, expert_done,
                                  is_expert, use_target, OFFLINE_METHOD_LOSS,
                                  method_regularize, method_div)
    q2_loss, loss_dict2 = iq_loss(agent, current_Q[1], vec_v_args,
                                  vec_next_v_args, vec_actions, expert_done,
                                  is_expert, use_target, OFFLINE_METHOD_LOSS,
                                  method_regularize, method_div)
    critic_loss = 1 / 2 * (q1_loss + q2_loss)
    # merge loss dicts
    loss_dict = average_dicts(loss_dict1, loss_dict2)
  else:
    critic_loss, loss_dict = iq_loss(agent, current_Q, vec_v_args,
                                     vec_next_v_args, vec_actions, expert_done,
                                     is_expert, use_target, OFFLINE_METHOD_LOSS,
                                     method_regularize, method_div)

  # logger.log('train/critic_loss', critic_loss, step)

  # Optimize the critic
  self.critic_optimizer.zero_grad()
  critic_loss.backward()
  if hasattr(self, 'clip_grad_val') and self.clip_grad_val:
    nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
  # step critic
  self.critic_optimizer.step()

  # args
  vdice_actor = False
  num_actor_updates = 1

  if self.actor and update_count % self.actor_update_frequency == 0:
    if not vdice_actor:

      obs = expert_batch[0]

      if num_actor_updates:
        for i in range(num_actor_updates):
          actor_alpha_losses = self.update_actor_and_alpha(
              obs, logger, update_count)

      loss_dict.update(actor_alpha_losses)

  if use_target and update_count % self.critic_target_update_frequency == 0:
    if do_soft_update:
      soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
    else:
      hard_update(self.critic_net, self.critic_target_net)
  return loss_dict
