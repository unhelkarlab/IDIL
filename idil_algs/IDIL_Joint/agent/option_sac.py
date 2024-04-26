import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch.optim import Adam
from idil_algs.baselines.IQLearn.utils.utils import (soft_update, one_hot,
                                                     one_hot_w_nan)
from .option_models import AbstractOptionActor, AbstractOptionThinker
from omegaconf import DictConfig


class OptionSAC(object):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim,
               discrete_obs, critic: nn.Module, actor: AbstractOptionActor,
               thinker: AbstractOptionThinker):
    self.gamma = config.gamma
    self.batch_size = config.mini_batch_size
    self.discrete_obs = discrete_obs
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim

    self.device = torch.device(config.device)

    self.init_temp = config.init_temp
    self.critic_tau = 0.005
    self.learn_temp = config.learn_temp
    self.actor_update_frequency = 1
    self.critic_target_update_frequency = 1

    self.num_critic_update = config.num_critic_update
    self.num_actor_update = config.num_actor_update
    self.clip_grad_val = config.clip_grad_val
    self.thinker_clip_grad_val = config.thinker_clip_grad_val

    self._critic = critic.to(self.device)

    self.critic_target = copy.deepcopy(self._critic).to(self.device)
    self.critic_target.load_state_dict(self._critic.state_dict())

    self.actor = actor.to(self.device)
    self.thinker = thinker.to(self.device)

    self.log_alpha = torch.tensor(np.log(self.init_temp)).to(self.device)
    self.log_alpha.requires_grad = True
    # Target Entropy = −dim(A)
    self.target_entropy = -action_dim

    # prev latent, prev action
    self.extra_action_dim = config.extra_action_dim
    self.extra_option_dim = config.extra_option_dim
    self.use_prev_action = config.use_prev_action
    if self.use_prev_action is True:
      raise NotImplementedError("use_prev_action==True is not implemented yet")

    # separate policy update
    self.separate_policy_update = config.separate_policy_update

    NAN = float("nan")
    self.PREV_LATENT = lat_dim
    if self.extra_action_dim:
      self.PREV_ACTION = (NAN if self.actor.is_discrete() else np.full(
          self.action_dim, NAN, dtype=float))
    else:
      self.PREV_ACTION = (NAN if self.actor.is_discrete() else np.zeros(
          self.action_dim, dtype=float))

    # optimizers
    self.reset_optimizers(config)

    self.train()
    self.critic_target.train()

  def reset_optimizers(self, config):
    thinker_betas = critic_betas = alpha_betas = actor_betas = [0.9, 0.999]
    self.actor_optimizer = Adam(self.actor.parameters(),
                                lr=config.optimizer_lr_policy,
                                betas=actor_betas)
    self.thinker_optimizer = Adam(self.thinker.parameters(),
                                  lr=config.optimizer_lr_option,
                                  betas=thinker_betas)
    self.critic_optimizer = Adam(self._critic.parameters(),
                                 lr=config.optimizer_lr_critic,
                                 betas=critic_betas)
    self.log_alpha_optimizer = Adam([self.log_alpha],
                                    lr=config.optimizer_lr_alpha,
                                    betas=alpha_betas)

  def train(self, training=True):
    self.training = training
    self.actor.train(training)
    self.thinker.train(training)
    self._critic.train(training)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  def critic_net(self):
    return self._critic

  @property
  def critic_target_net(self):
    return self.critic_target

  def conv_input(self,
                 batch_input,
                 is_onehot_needed,
                 dimension,
                 extra_dim=False):
    if is_onehot_needed:
      if not isinstance(batch_input, torch.Tensor):
        batch_input = torch.tensor(
            batch_input, dtype=torch.float).reshape(-1).to(self.device)
      else:
        batch_input = batch_input.reshape(-1)
      dimension = dimension + int(extra_dim)
      batch_input = one_hot(batch_input, dimension)
    else:
      if not isinstance(batch_input, torch.Tensor):
        batch_input = torch.tensor(np.array(batch_input).reshape(-1, dimension),
                                   dtype=torch.float).to(self.device)

    return batch_input

  # def conv_input(self, batch_input, is_discrete, dimension, extra_dim=False):
  #   if extra_dim:
  #     # find nan
  #     if is_discrete:
  #       batch_input = torch.tensor(
  #           batch_input, dtype=torch.float).reshape(-1).to(self.device)
  #       mask_nan = batch_input.isnan()
  #       non_nan_input = batch_input[~mask_nan]

  #       n_batch = len(batch_input)
  #       batch_conv = torch.zeros((n_batch, dimension + 1),
  #                                dtype=torch.float).to(device=self.device)
  #       if len(non_nan_input) != 0:
  #         batch_conv[~mask_nan, :-1] = one_hot(non_nan_input, dimension)

  #       batch_conv[mask_nan, -1] = 1.0
  #     else:
  #       if not isinstance(batch_input, torch.Tensor):
  #         batch_input = torch.tensor(batch_input,
  #                                    dtype=torch.float).to(self.device)
  #         if batch_input.ndim < 2:
  #           batch_input = batch_input.unsqueeze(0)
  #       mask_nan = batch_input[:, 0].isnan()
  #       batch_input[mask_nan, :] = 0.0
  #       is_prev = torch.zeros(len(batch_input)).to(device=self.device)
  #       is_prev[mask_nan] = 1.0
  #       batch_conv = torch.cat([batch_input, is_prev.reshape(-1, 1)], dim=-1)

  #     return batch_conv
  #   else:
  #     if is_discrete:
  #       batch_input = torch.tensor(
  #           batch_input, dtype=torch.float).reshape(-1).to(self.device)
  #       batch_input = one_hot_w_nan(batch_input, dimension)
  #     else:
  #       if not isinstance(batch_input, torch.Tensor):
  #         batch_input = torch.tensor(batch_input,
  #                                    dtype=torch.float).to(self.device)
  #         if batch_input.ndim < 2:
  #           batch_input = batch_input.unsqueeze(0)

  #     return batch_input

  def gather_mental_probs(self, state, prev_latent, prev_action):
    # --- convert inputs
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)
    prev_latent = self.conv_input(prev_latent, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)
    # prev_action = self.conv_input(prev_action, self.actor.is_discrete(),
    #                               self.action_dim, self.extra_action_dim)
    prev_action = None

    with torch.no_grad():
      probs, log_probs = self.thinker.mental_probs(state, prev_latent,
                                                   prev_action)

    return probs.cpu().detach().numpy(), log_probs.cpu().detach().numpy()

  def evaluate_action(self, state, latent, action):
    # --- convert inputs
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)
    latent = self.conv_input(latent, self.thinker.is_discrete(), self.lat_dim)

    # --- action
    if not isinstance(action, torch.Tensor):
      n_col = 1 if self.actor.is_discrete() else self.action_dim
      action = torch.tensor(np.array(action).reshape(-1, n_col)).to(self.device)
    else:
      if action.ndim < 2:
        action = action.unsqueeze(0)

    with torch.no_grad():
      log_prob = self.actor.evaluate_action(state, latent, action)
    return log_prob.cpu().detach().numpy()

  def infer_mental_states(self, state, action):
    len_demo = len(state)

    # --- convert inputs
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)

    lat_indices = np.arange(self.lat_dim)
    latent = self.conv_input(lat_indices, self.thinker.is_discrete(),
                             self.lat_dim)
    prev_latent = self.conv_input(lat_indices, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)
    prev_latent0 = self.conv_input(np.array([self.PREV_LATENT]),
                                   self.thinker.is_discrete(), self.lat_dim,
                                   self.extra_option_dim)

    # --- action
    if not isinstance(action, torch.Tensor):
      n_col = 1 if self.actor.is_discrete() else self.action_dim
      action = torch.tensor(np.array(action).reshape(-1, n_col)).to(self.device)
    else:
      if action.ndim < 2:
        action = action.unsqueeze(0)

    state = state.repeat_interleave(self.lat_dim, dim=0)
    action = action.repeat_interleave(self.lat_dim, dim=0)
    latent = latent.repeat(len_demo, 1)
    prev_latent = prev_latent.repeat(len_demo, 1)

    with torch.no_grad():
      log_pis = self.actor.evaluate_action(state, latent, action)
      log_pis = log_pis.view(-1, 1, self.lat_dim)

      _, log_trs = self.thinker.mental_probs(state, prev_latent, None)
      log_trs = log_trs.view(-1, self.lat_dim, self.lat_dim)
      log_prob = log_trs + log_pis
      _, log_tr0 = self.thinker.mental_probs(state[0], prev_latent0[0], None)
      log_prob0 = log_tr0[-1] + log_pis[0, 0]

      # forward
      max_path = torch.empty(len_demo,
                             self.lat_dim,
                             dtype=torch.long,
                             device=self.device)
      accumulate_logp = log_prob0
      max_path[0] = self.lat_dim
      for i in range(1, len_demo):
        accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) +
                                           log_prob[i]).max(dim=-2)
      # backward
      c_array = torch.zeros(len_demo + 1,
                            1,
                            dtype=torch.long,
                            device=self.device)
      log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
      for i in range(len_demo, 0, -1):
        c_array[i - 1] = max_path[i - 1][c_array[i]]

    return (c_array[1:].detach().cpu().numpy(),
            log_prob_traj.detach().cpu().numpy())

  def choose_action(self, state, prev_latent, prev_action, sample=False):
    # --- convert inputs
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)
    prev_latent = self.conv_input(prev_latent, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)
    prev_action = None
    if self.use_prev_action:
      prev_action = self.conv_input(prev_action, self.actor.is_discrete(),
                                    self.action_dim, self.extra_action_dim)

    with torch.no_grad():
      if sample:
        latent, _ = self.thinker.sample(state, prev_latent, prev_action)
        latent_item = latent.detach().cpu().numpy()[0]
        if self.thinker.is_discrete():
          latent = one_hot(latent, self.lat_dim)

        action, _ = self.actor.sample(state, latent)
        action_item = action.detach().cpu().numpy()[0]
      else:
        # latent = self.thinker.exploit(state, prev_latent, prev_action)
        latent, _ = self.thinker.sample(state, prev_latent, prev_action)
        latent_item = latent.detach().cpu().numpy()[0]
        if self.thinker.is_discrete():
          latent = one_hot(latent, self.lat_dim)

        if self.actor.is_discrete():
          action, _ = self.actor.sample(state, latent)
        else:
          action = self.actor.exploit(state, latent)
        action_item = action.detach().cpu().numpy()[0]

    return latent_item, action_item

  def choose_policy_action(self, state, latent, sample=False):
    # --- convert inputs
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)
    latent = self.conv_input(latent, self.thinker.is_discrete(), self.lat_dim)

    with torch.no_grad():
      if sample or self.actor.is_discrete():
        action, _ = self.actor.sample(state, latent)
      else:
        action = self.actor.exploit(state, latent)

    return action.detach().cpu().numpy()[0]

  def choose_mental_state(self, state, prev_latent, sample=False):
    # --- convert inputs
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)
    prev_latent = self.conv_input(prev_latent, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)

    prev_action = None
    with torch.no_grad():
      latent, _ = self.thinker.sample(state, prev_latent, prev_action)

    return latent.detach().cpu().numpy()[0]

  def critic(self, obs, prev_latent, prev_action, latent, action, both=False):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    prev_latent = self.conv_input(prev_latent, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)
    # --- convert latent
    if self.thinker.is_discrete():
      latent = one_hot(latent, self.lat_dim)
    # ------

    prev_action = None
    if self.use_prev_action:
      prev_action = self.conv_input(prev_action, self.actor.is_discrete(),
                                    self.action_dim, self.extra_action_dim)
    # --- convert discrete action
    if self.actor.is_discrete():
      action = one_hot(action, self.action_dim)
    # ------

    return self._critic(obs, prev_latent, prev_action, latent, action, both)

  def getV(self, obs, prev_latent, prev_action):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    # --- convert prev_latent
    prev_latent = self.conv_input(prev_latent, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)
    # ------

    # --- convert prev_action
    prev_action = None
    if self.use_prev_action:
      prev_action = self.conv_input(prev_action, self.actor.is_discrete(),
                                    self.action_dim, self.extra_action_dim)
    # ------

    latent, lat_log_prob = self.thinker.sample(obs, prev_latent, prev_action)
    # --- convert prev_latent
    if self.thinker.is_discrete():
      latent = one_hot(latent, self.lat_dim)
    # ------

    action, act_log_prob = self.actor.sample(obs, latent)
    # --- convert discrete action
    if self.actor.is_discrete():
      action = one_hot(action, self.action_dim)
    # ------

    current_Q = self._critic(obs, prev_latent, prev_action, latent, action)
    current_V = current_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
    return current_V

  def get_targetV(self, obs, prev_latent, prev_action):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    # --- convert prev_latent
    prev_latent = self.conv_input(prev_latent, self.thinker.is_discrete(),
                                  self.lat_dim, self.extra_option_dim)
    # ------

    # --- convert prev_action
    prev_action = None
    if self.use_prev_action:
      prev_action = self.conv_input(prev_action, self.actor.is_discrete(),
                                    self.action_dim, self.extra_action_dim)
    # ------

    latent, lat_log_prob = self.thinker.sample(obs, prev_latent, prev_action)
    # --- convert prev_latent
    if self.thinker.is_discrete():
      latent = one_hot(latent, self.lat_dim)
    # ------

    action, act_log_prob = self.actor.sample(obs, latent)
    # --- convert discrete action
    if self.actor.is_discrete():
      action = one_hot(action, self.action_dim)
    # ------

    current_Q = self.critic_target(obs, prev_latent, prev_action, latent,
                                   action)
    current_V = current_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
    return current_V

  def update(self, replay_buffer, logger, step):
    (obs, prev_lat, prev_act, next_obs, latent, action, reward,
     done) = replay_buffer.get_samples(self.batch_size, self.device)

    losses = self.update_critic(obs, prev_lat, prev_act, next_obs, latent,
                                action, reward, done, logger, step)
    if step % self.actor_update_frequency == 0:
      actor_alpha_losses = self.update_actor_and_alpha(obs, prev_lat, prev_act,
                                                       logger, step)
      losses.update(actor_alpha_losses)

    # NOTE: ----
    if step % self.critic_target_update_frequency == 0:
      soft_update(self._critic, self.critic_target, self.critic_tau)

    return losses

  def update_critic(self, obs, prev_lat, prev_act, next_obs, latent, action,
                    reward, done, logger, step):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
      next_obs = one_hot(next_obs, self.obs_dim)
    # ------

    # --- convert latent
    prev_lat = self.conv_input(prev_lat, self.thinker.is_discrete(),
                               self.lat_dim, self.extra_option_dim)
    if self.thinker.is_discrete():
      latent = one_hot(latent, self.lat_dim)
    # ------

    # --- convert action
    prev_act = None
    if self.use_prev_action:
      prev_act = self.conv_input(prev_act, self.actor.is_discrete(),
                                 self.action_dim, self.extra_action_dim)
    if self.actor.is_discrete():
      action = one_hot(action, self.action_dim)
    # ------

    # --- convert action
    zero_column = torch.zeros(len(latent)).reshape(-1, 1).to(device=self.device)
    if self.extra_option_dim:
      latent_ = torch.cat([latent, zero_column], dim=-1)
    else:
      latent_ = latent

    if self.extra_action_dim:
      action_ = torch.cat([action, zero_column], dim=-1)
    else:
      action_ = action

    # ------
    with torch.no_grad():
      next_latent, lat_log_prob = self.thinker.sample(next_obs, latent_,
                                                      action_)
      if self.thinker.is_discrete():
        next_latent = one_hot(next_latent, self.lat_dim)

      next_action, act_log_prob = self.actor.sample(next_obs, next_latent)
      if self.actor.is_discrete():
        next_action = one_hot(next_action, self.action_dim)

      target_Q = self.critic_target(next_obs, latent_, action_, next_latent,
                                    next_action)
      target_V = target_Q - self.alpha.detach() * (act_log_prob + lat_log_prob)
      target_Q = reward + (1 - done) * self.gamma * target_V

    # get current Q estimates
    current_Q = self._critic(obs, prev_lat, prev_act, latent, action, both=True)
    if isinstance(current_Q, tuple):
      q1_loss = F.mse_loss(current_Q[0], target_Q)
      q2_loss = F.mse_loss(current_Q[1], target_Q)
      critic_loss = q1_loss + q2_loss
    else:
      critic_loss = F.mse_loss(current_Q, target_Q)

    # logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if self.clip_grad_val:
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
    self.critic_optimizer.step()

    return {'loss/critic': critic_loss.item()}

  def update_actor_and_alpha(self, obs, prev_lat, prev_act, logger, step):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    prev_lat = self.conv_input(prev_lat, self.thinker.is_discrete(),
                               self.lat_dim, self.extra_option_dim)
    prev_act = None
    if self.use_prev_action:
      prev_act = self.conv_input(prev_act, self.actor.is_discrete(),
                                 self.action_dim, self.extra_action_dim)

    if self.separate_policy_update:
      # thinker update
      latent, lat_log_prob = self.thinker.rsample(obs, prev_lat, prev_act)
      action, act_log_prob = self.actor.rsample(obs, latent)
      actor_Q = self._critic(obs, prev_lat, prev_act, latent, action)

      option_loss = (self.alpha.detach() * (act_log_prob + lat_log_prob) -
                     actor_Q).mean()
      self.thinker_optimizer.zero_grad()
      option_loss.backward()
      if self.thinker_clip_grad_val:
        nn.utils.clip_grad_norm_(self.thinker.parameters(),
                                 self.thinker_clip_grad_val)
      self.thinker_optimizer.step()

      # actor update
      latent, lat_log_prob = self.thinker.rsample(obs, prev_lat, prev_act)
      action, act_log_prob = self.actor.rsample(obs, latent)
      actor_Q = self._critic(obs, prev_lat, prev_act, latent, action)

      actor_loss = (self.alpha.detach() * (act_log_prob + lat_log_prob) -
                    actor_Q).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      if self.clip_grad_val:
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_val)
      self.actor_optimizer.step()

      losses = {
          'loss/actor': actor_loss.item(),
          'loss/thinker': option_loss.item(),
          'actor_loss/actor_entropy': -act_log_prob.mean().item(),
          'actor_loss/thinker_entropy': -lat_log_prob.mean().item()
      }
    else:
      latent, lat_log_prob = self.thinker.rsample(obs, prev_lat, prev_act)
      action, act_log_prob = self.actor.rsample(obs, latent)

      actor_Q = self._critic(obs, prev_lat, prev_act, latent, action)

      actor_loss = (self.alpha.detach() * (act_log_prob + lat_log_prob) -
                    actor_Q).mean()

      # logger.log('train/actor_loss', actor_loss, step)
      # logger.log('train/actor_entropy', -act_log_prob.mean(), step)
      # logger.log('train/thinker_entropy', -lat_log_prob.mean(), step)

      # optimize the actor
      self.actor_optimizer.zero_grad()
      self.thinker_optimizer.zero_grad()
      actor_loss.backward()
      if self.clip_grad_val:
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_val)
      if self.thinker_clip_grad_val:
        nn.utils.clip_grad_norm_(self.thinker.parameters(),
                                 self.thinker_clip_grad_val)
      self.actor_optimizer.step()
      self.thinker_optimizer.step()

      losses = {
          'loss/actor': actor_loss.item(),
          'actor_loss/actor_entropy': -act_log_prob.mean().item(),
          'actor_loss/thinker_entropy': -lat_log_prob.mean().item()
      }

    # TODO: implement learn alpha
    # if self.learn_temp:
    #   self.log_alpha_optimizer.zero_grad()
    #   alpha_loss = (self.log_alpha *
    #                 (-log_prob - self.target_entropy).detach()).mean()
    #   logger.log('train/alpha_loss', alpha_loss, step)
    #   logger.log('train/alpha_value', self.alpha, step)

    #   alpha_loss.backward()
    #   self.log_alpha_optimizer.step()

    #   losses.update({
    #       'alpha_loss/loss': alpha_loss.item(),
    #       'alpha_loss/value': self.alpha.item(),
    #   })
    return losses

  # Save model parameters
  def save(self, path, suffix=""):
    actor_path = f"{path}{suffix}_actor"
    thinker_path = f"{path}{suffix}_thinker"
    critic_path = f"{path}{suffix}_critic"

    # print('Saving models to {} and {}'.format(actor_path, critic_path))
    torch.save(self.actor.state_dict(), actor_path)
    torch.save(self.thinker.state_dict(), thinker_path)
    torch.save(self._critic.state_dict(), critic_path)

  # Load model parameters
  def load(self, path):
    actor_path = f'{path}_actor'
    thinker_path = f'{path}_thinker'
    critic_path = f'{path}_critic'
    print('Loading models from {}, {} and {}'.format(actor_path, thinker_path,
                                                     critic_path))
    if actor_path is not None:
      self.actor.load_state_dict(
          torch.load(actor_path, map_location=self.device))
    if thinker_path is not None:
      self.thinker.load_state_dict(
          torch.load(thinker_path, map_location=self.device))
    if critic_path is not None:
      self._critic.load_state_dict(
          torch.load(critic_path, map_location=self.device))
