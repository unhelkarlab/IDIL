import os
import pickle
import random
from typing import List
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from idil_algs.baselines.IQLearn.utils.utils import (make_env, eval_mode,
                                                     compute_expert_return_mean)
from idil_algs.baselines.IQLearn.dataset.expert_dataset import (ExpertDataset)
from idil_algs.baselines.IQLearn.utils.logger import Logger
from idil_algs.IDIL_Joint.helper.option_memory import (OptionMemory)
from idil_algs.IDIL_Joint.helper.utils import (get_expert_batch, evaluate, save,
                                               get_samples, build_expert_batch_with_topK)
from .agent.make_agent import MentalIQL
from .agent.make_agent import make_miql_agent
import wandb
import omegaconf

# fixed pi impotr
from .utils import DiscreteExpertPolicySampler

def load_expert_data_w_labels(demo_path, num_trajs, n_labeled, seed):
  expert_dataset = ExpertDataset(demo_path, num_trajs, 1, seed + 42)
  print(f'--> Expert memory size: {len(expert_dataset)}')

  cnt_label = 0
  traj_labels = []
  for i_e in range(num_trajs):
    if "latents" in expert_dataset.trajectories:
      expert_latents = expert_dataset.trajectories["latents"][i_e]
    else:
      expert_latents = None

    if i_e < n_labeled:
      traj_labels.append(expert_latents)
      cnt_label += 1
    else:
      traj_labels.append(None)

  print(f"num_labeled: {cnt_label} / {num_trajs}, num_samples: ",
        len(expert_dataset))
  return expert_dataset, traj_labels, cnt_label


def get_top_k_trajectories(agent: MentalIQL,
                           trajectories,
                           k: float = 0.2,
                           randomize=False):
  """
  Given input trajectories, this function computes the top-k most uncertain
  by entropy, and returns an new object with keys matching the input trajectories object.
  """
  num_samples = len(trajectories["states"])

  traj_entropy_tuples = []

  for i_e in range(num_samples):
    expert_states = trajectories["states"][i_e]
    expert_actions = trajectories["actions"][i_e]

    _, _, entropy = agent.infer_mental_states(expert_states, expert_actions)
    traj_entropy_tuples.append((i_e, entropy))

  # sort the inferred mental arrays by entropy, in descending entropy order
  if randomize:
    random.shuffle(traj_entropy_tuples)
  else:
    traj_entropy_tuples = sorted(traj_entropy_tuples,
                                  key=lambda x: x[1],
                                  reverse=True)
    
  # build a new object with the top-k most uncertain trajectories
  output_trajs = {}
  max_num_trajs = int(k * num_samples)
  for key in trajectories.keys():
    # NOTE: this changes the index of the trajectories
    output_trajs[key] = [trajectories[key][i_e] for i_e, _ in traj_entropy_tuples[:max_num_trajs]]

  return output_trajs

def infer_mental_states_all_demo(agent: MentalIQL,
                                 expert_traj, 
                                 traj_labels,
                                 entropy_scoring: bool = False,
                                 k: float = 1.0):
  num_samples = len(expert_traj["states"])
  list_mental_states = []
  list_mental_states_idx = []
  inferred_mental_arrays = []

  for i_e in range(num_samples):
    if traj_labels[i_e] is None:
      expert_states = expert_traj["states"][i_e] # list of len expert_traj['lengths'][i_e]
      expert_actions = expert_traj["actions"][i_e] # list of len expert_traj['lengths'][i_e]
      
      # if expert intents are not available, we can infer them
      # using the intent policy model
      mental_array, _, entropy = agent.infer_mental_states(expert_states, expert_actions)
      inferred_mental_arrays.append((mental_array, i_e, entropy))

    else:
      # if expert intents are available, use them
      mental_array = traj_labels[i_e]
      # only append mental array if labels are available
      list_mental_states.append(mental_array)
      list_mental_states_idx.append(i_e)


  if entropy_scoring and k:
    # sort the inferred mental arrays by entropy, in descending entropy order
    inferred_mental_arrays = sorted(inferred_mental_arrays,
                                    key=lambda x: x[2],
                                    reverse=True)
    # select the top k mental arrays
    # _k = min(k, len(inferred_mental_arrays))
    _k = int(k * len(inferred_mental_arrays))
    inferred_mental_arrays = inferred_mental_arrays[:_k]

  _extend_mental_array = [mental_array for mental_array, _, _ in inferred_mental_arrays]
  _extend_mental_array_idx = [i_e for _, i_e, _ in inferred_mental_arrays]

  list_mental_states.extend(_extend_mental_array)
  list_mental_states_idx.extend(_extend_mental_array_idx)

  return list_mental_states, list_mental_states_idx


def infer_last_next_mental_state(agent: MentalIQL, expert_traj,
                                 list_mental_states,
                                 list_mental_states_idx: List[int],
                                 k: int = None):
  list_last_next_mental_state = []
  for mental_idx, traj_idx in enumerate(list_mental_states_idx):
    last_next_state = expert_traj["next_states"][traj_idx][-1]
    last_mental_state = list_mental_states[mental_idx][-1]

    last_next_mental_state = agent.choose_mental_state(last_next_state,
                                                       last_mental_state, False)
    list_last_next_mental_state.append(last_next_mental_state)

  return list_last_next_mental_state


def infer_next_latent(agent: MentalIQL, trajectories):
  """
  Given an object with keys including 'states' and 'latents', this function
  infers the next-to-last intent for each trajectory in the input object.

  Returns:
    - list of lists where each inner element is the sequence of next-latents (size num_states - 1)
    extended with the extra, next-to-last latent
  """
  num_samples = len(trajectories["states"])
  list_next_latents = []
  for i_e in range(num_samples):
    last_next_state = trajectories["next_states"][i_e][-1]
    latents = trajectories["latents"][i_e]
    last_latent = latents[-1]

    next_latent = agent.choose_mental_state(last_next_state, last_latent, False)
    assert isinstance(latents, list), f"latents is not a list, but {type(latents)}"
    next_latents = latents + [next_latent]

    list_next_latents.append(next_latents)
  
  return list_next_latents


def train(config: omegaconf.DictConfig,
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
  replay_mem = config.n_sample
  max_explore_step = config.max_explore_step
  eps_window = 10
  num_episodes = 8

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  run_name = f"{config.alg_name}_{config.tag}"
  wandb.init(project=f"{os.environ.get('WANDB_PROJECT')}-{env_name}",
             name=run_name,
             entity=os.environ.get("WANDB_ENTITY"),
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  alg_type = 'iq'

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
  max_explore_step = int(max_explore_step)

  # Load expert data
  n_labeled = int(num_trajs * config.supervision)
  expert_dataset, traj_labels, cnt_label = load_expert_data_w_labels(
      demo_path, num_trajs, n_labeled, seed)

  expert_avg, expert_std = compute_expert_return_mean(
      expert_dataset.trajectories)
  
  # ---- load the extra trajectories
  with open(config.extra_trajectories_path, 'rb') as f:
    extra_trajectories = pickle.load(f)
  print(f"\nLoaded {len(extra_trajectories['states'])} extra trajectories with keys: {list(extra_trajectories.keys())}\n\n")

  # ---- initialize wandb metrics
  wandb.run.summary["expert_avg"] = expert_avg
  wandb.run.summary["expert_std"] = expert_std


  # ---- initialize agent, fixed pi implementation
  if config.fixed_pi:
    agent = make_miql_agent(config, env, fixed_pi=config.fixed_pi,
                            expert_dataset=expert_dataset)

  else:
    agent = make_miql_agent(config, env) 

  output_suffix = f"_n{num_trajs}_l{cnt_label}"
  online_memory_replay = OptionMemory(replay_mem, seed + 1)

  batch_size = min(batch_size, len(expert_dataset))

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
  expert_data = None

  for epoch in count():
    episode_reward = 0
    done = False

    state = env.reset()
    prev_lat, prev_act = agent.PREV_LATENT, agent.PREV_ACTION
    latent = agent.choose_mental_state(state, prev_lat, sample=True)

    for episode_step in count():
      with eval_mode(agent):
        # sampling action from policy, use action to infer next state and then latent
        # this action will be added to the replay buffer
        action = agent.choose_policy_action(state, latent, sample=True)

        next_state, reward, done, info = env.step(action)
        next_latent = agent.choose_mental_state(next_state, latent, sample=True)

      episode_reward += reward

      if explore_steps % eval_interval == 0 and begin_learn:
        eval_returns, eval_timesteps, successes = evaluate(
            agent, eval_env, num_episodes=num_episodes)
        returns = np.mean(eval_returns)
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

      # only store done true when episode finishes without hitting timelimit
      done_no_lim = done
      if info.get('TimeLimit.truncated', False):
        done_no_lim = 0
      online_memory_replay.add((prev_lat, prev_act, state, latent, action,
                                next_state, next_latent, reward, done_no_lim))

      explore_steps += 1
      if online_memory_replay.size() >= initial_mem:
        # Start learning
        if begin_learn is False:
          print('Learn begins!')
          begin_learn = True

        if explore_steps == max_explore_step:
          print('Finished!')
          wandb.finish()
          return

        # ##### sample batch
        # infer mental states of expert data
        if (expert_data is None
            or explore_steps % config.demo_latent_infer_interval == 0):

          # --- here begings inference + data formatting ---
          top_k_trajectories = get_top_k_trajectories(agent, 
                                                      extra_trajectories, 
                                                      k=config.k,
                                                      randomize=config.randomize)
          expert_next_latents = infer_next_latent(agent, expert_dataset.trajectories)
          extra_next_latents = infer_next_latent(agent, top_k_trajectories)
          exb = build_expert_batch_with_topK(expert_trajectories=expert_dataset.trajectories,
                                             extra_trajectories=top_k_trajectories,
                                             device= agent.device,
                                             init_latent=agent.PREV_LATENT,
                                             init_action=agent.PREV_ACTION,
                                             expert_next_latents=expert_next_latents,
                                             extra_next_latents=extra_next_latents)
          
          expert_data = (exb["prev_latents"], exb["prev_actions"],
                         exb["states"], exb["latents"], exb["actions"],
                         exb["next_states"], exb["next_latents"],
                         exb["rewards"], exb["dones"])
        # ##### end of batch sampling


        ######
        # IQ-Learn
        tx_losses = pi_losses = {}
        if explore_steps % config.update_interval == 0:
          policy_batch = online_memory_replay.get_samples(
              batch_size, agent.device)
          
          # expert batch is the expert distribution
          expert_batch = get_samples(batch_size, expert_data)

          tx_losses, pi_losses = agent.miql_update(
              policy_batch, expert_batch, config.demo_latent_infer_interval,
              logger, explore_steps)

        if explore_steps % log_interval == 0:
          for key, loss in tx_losses.items():
            writer.add_scalar("tx_loss/" + key, loss, global_step=explore_steps)
          if not config.fixed_pi:
            for key, loss in pi_losses.items():
              writer.add_scalar("pi_loss/" + key, loss, global_step=explore_steps)

      if done:
        break
      state = next_state
      prev_lat = latent
      prev_act = action
      latent = next_latent

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