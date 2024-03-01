import os
import random
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from ml_algs.baselines.IQLearn.utils.utils import (make_env, eval_mode,
                                                   compute_expert_return_mean)
from ml_algs.baselines.IQLearn.utils.logger import Logger
from .agent.make_agent import make_oiql_agent
from .helper.option_memory import OptionMemory
from .helper.utils import get_expert_batch, evaluate, save, get_samples
from ml_algs.IDIL.train import (load_expert_data_w_labels,
                                infer_mental_states_all_demo)
import wandb
import omegaconf


def trainer(config: omegaconf.DictConfig,
            demo_path,
            num_trajs,
            log_dir,
            output_dir,
            log_interval=500,
            eval_interval=5000,
            env_kwargs={}):

  env_name = config.env_name
  seed = config.seed
  batch_size = config.mini_batch_size
  num_latent = config.dim_c
  replay_mem = config.n_sample
  num_explore_steps = config.max_explore_step
  output_suffix = ""
  load_path = None
  eps_window = 10

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

  alg_type = "rl"
  imitation = True
  fn_make_agent = make_oiql_agent
  alg_type = 'iq'

  # constants
  num_episodes = 8

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

  initial_mem = int(config.init_sample)
  replay_mem = int(replay_mem)
  assert initial_mem <= replay_mem
  eps_window = int(eps_window)
  num_explore_steps = int(num_explore_steps)

  use_target = True
  do_soft_update = True
  agent = fn_make_agent(config, env)

  if load_path is not None:
    if os.path.isfile(load_path):
      print("=> loading pretrain '{}'".format(load_path))
      agent.load(load_path)
    else:
      print("[Attention]: Did not find checkpoint {}".format(load_path))

  if imitation:
    # # Load expert data
    # expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
    # print(f'--> Expert memory size: {len(expert_dataset)}')
    n_labeled = int(num_trajs * config.supervision)
    expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
        demo_path, num_trajs, n_labeled, seed)
    output_suffix = f"_n{num_trajs}_l{cnt_label}"
    batch_size = min(batch_size, len(expert_dataset))

    expert_avg, expert_std = compute_expert_return_mean(
        expert_dataset.trajectories)

    wandb.run.summary["expert_avg"] = expert_avg
    wandb.run.summary["expert_std"] = expert_std

  online_memory_replay = OptionMemory(replay_mem, seed + 1)

  # Setup logging
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  run_name=f"{env_name}_{run_name}")

  # track mean reward and scores
  best_eval_returns = -np.inf
  best_success_rate = -np.inf
  rewards_window = deque(maxlen=eps_window)  # last N rewards
  epi_step_window = deque(maxlen=eps_window)
  success_window = deque(maxlen=eps_window)
  cnt_steps = 0

  begin_learn = False
  episode_reward = 0
  explore_steps = 0
  update_count = 0
  expert_data = None

  for epoch in count():
    state = env.reset()
    prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    episode_reward = 0
    done = False

    for episode_step in count():
      with eval_mode(agent):
        # if not begin_learn:
        #   action = env.action_space.sample()
        # else:
        latent, action = agent.choose_action(state,
                                             prev_lat,
                                             prev_act,
                                             sample=True)

      next_state, reward, done, info = env.step(action)
      episode_reward += reward

      if explore_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps, successes = evaluate(
            agent, eval_env, num_episodes=num_episodes)
        returns = np.mean(eval_returns)
        # explore_steps += 1  # To prevent repeated eval at timestep 0
        # logger.log('eval/episode', epoch, explore_steps)
        logger.log('eval/episode_reward', returns, explore_steps)
        logger.log('eval/episode_step', np.mean(eval_timesteps), explore_steps)
        if len(successes) > 0:
          success_rate = np.mean(successes)
          logger.log('eval/success_rate', success_rate, explore_steps)
          if success_rate > best_success_rate:
            best_success_rate = success_rate
            wandb.run.summary["best_success_rate"] = best_success_rate

        logger.dump(explore_steps, ty='eval')

        if returns >= best_eval_returns:
          # Store best eval returns
          best_eval_returns = returns
          wandb.run.summary["best_returns"] = best_eval_returns
          save(agent,
               epoch,
               1,
               env_name,
               alg_type,
               output_dir=output_dir,
               suffix=output_suffix + "_best")
        # else:
        #   # for temporary use
        #   save(agent,
        #        epoch,
        #        1,
        #        env_name,
        #        alg_type,
        #        output_dir=output_dir,
        #        suffix=output_suffix + f"_{epoch}")

      # only store done true when episode finishes without hitting timelimit
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((state, prev_lat, prev_act, next_state, latent,
                                action, reward, done_no_lim))

      explore_steps += 1
      if online_memory_replay.size() >= initial_mem:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        if explore_steps == num_explore_steps:
          print('Finished!')
          wandb.finish()
          return

          # ##### sample batch
          # infer mental states of expert data
        if imitation and (expert_data is None or explore_steps %
                          config.demo_latent_infer_interval == 0):
          inferred_latents = infer_mental_states_all_demo(
              agent, expert_dataset.trajectories, traj_labels)
          exb = get_expert_batch(expert_dataset.trajectories, inferred_latents,
                                 agent.device, agent.PREV_LATENT,
                                 agent.PREV_ACTION)
          expert_data = (exb["states"], exb["prev_latents"],
                         exb["prev_actions"], exb["next_states"],
                         exb["latents"], exb["actions"], exb["rewards"],
                         exb["dones"])

          ######
          # IQ-Learn Modification
        losses = {}
        if explore_steps % config.update_interval == 0:
          if imitation:
            expert_batch = get_samples(batch_size, expert_data)
            policy_batch = online_memory_replay.get_samples(
                batch_size, agent.device)
            losses = agent.iq_update(policy_batch, expert_batch, logger,
                                     update_count, use_target, do_soft_update,
                                     config.method_loss,
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
      prev_lat = latent
      prev_act = action

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
