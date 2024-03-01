from typing import Type
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from ml_algs.baselines.IQLearn.utils.utils import (one_hot, soft_update)
from .nn_models import AbstractOptionActor
from .option_abstract import AbstractPolicyLeaner
from omegaconf import DictConfig


class OptionSAC(AbstractPolicyLeaner):

  def __init__(self, config: DictConfig, obs_dim, action_dim, option_dim,
               discrete_obs, critic_base: Type[nn.Module],
               actor: AbstractOptionActor):
    super().__init__(config)
    self.discrete_obs = discrete_obs
    self.obs_dim = obs_dim
    self.action_dim = action_dim

    use_tanh = False
    self.init_temp = config.init_temp
    self.critic_tau = 0.005
    self.learn_temp = config.learn_temp
    self.actor_update_frequency = 1
    self.critic_target_update_frequency = 1

    self.num_critic_update = config.num_critic_update
    self.num_actor_update = config.num_actor_update
    self.clip_grad_val = config.clip_grad_val

    self._critic = critic_base(obs_dim, action_dim, option_dim,
                               config.hidden_critic, config.activation,
                               self.gamma, use_tanh).to(self.device)

    self.critic_target = critic_base(obs_dim, action_dim, option_dim,
                                     config.hidden_critic, config.activation,
                                     self.gamma, use_tanh).to(self.device)

    self.critic_target.load_state_dict(self._critic.state_dict())

    self.actor = actor.to(self.device)

    self.log_alpha = torch.tensor(np.log(self.init_temp)).to(self.device)
    self.log_alpha.requires_grad = True
    # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
    self.target_entropy = -action_dim

    # optimizers
    self.optimizer_lr_policy = config.optimizer_lr_policy
    self.optimizer_lr_critic = config.optimizer_lr_critic
    self.optimizer_lr_alpha = config.optimizer_lr_alpha
    self.reset_optimizers()

    self.train()
    self.critic_target.train()

  def train(self, training=True):
    self.training = training
    self.actor.train(training)
    self._critic.train(training)

  def reset_optimizers(self):
    actor_betas = critic_betas = alpha_betas = [0.9, 0.999]
    self.actor_optimizer = Adam(self.actor.parameters(),
                                lr=self.optimizer_lr_policy,
                                betas=actor_betas)
    self.critic_optimizer = Adam(self._critic.parameters(),
                                 lr=self.optimizer_lr_critic,
                                 betas=critic_betas)
    self.log_alpha_optimizer = Adam([self.log_alpha],
                                    lr=self.optimizer_lr_alpha,
                                    betas=alpha_betas)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  def critic_net(self):
    return self._critic

  @property
  def critic_target_net(self):
    return self.critic_target

  def choose_action(self, obs, option, sample=False):
    # --- convert state
    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    with torch.no_grad():
      if sample:
        action, _ = self.actor.sample(obs, option)
      else:
        action = self.actor.exploit(obs, option)

    return action.detach().cpu().numpy()[0]

  def critic(self, obs, option, action, both=False):
    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    action = self.conv_input(action, self.actor.is_discrete(), self.action_dim)
    option = self.conv_input(option, False, 1)

    return self._critic(obs, option, action, both)

  def getV(self, obs, option):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    action, log_prob = self.actor.sample(obs, option)
    action = self.conv_input(action, self.actor.is_discrete(), self.action_dim)

    current_Q = self._critic(obs, option, action)
    current_V = current_Q - self.alpha.detach() * log_prob
    return current_V

  def get_targetV(self, obs, option):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    action, log_prob = self.actor.sample(obs, option)
    action = self.conv_input(action, self.actor.is_discrete(), self.action_dim)

    target_Q = self.critic_target(obs, option, action)
    target_V = target_Q - self.alpha.detach() * log_prob
    return target_V

  def update(self, obs, option, action, next_obs, next_option, reward, done,
             logger, step):

    losses = self.update_critic(obs, option, action, next_obs, next_option,
                                reward, done, logger, step)

    if step % self.actor_update_frequency == 0:
      actor_alpha_losses = self.update_actor_and_alpha(obs, option, logger,
                                                       step)
      losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
      soft_update(self._critic, self.critic_target, self.critic_tau)

    return losses

  def update_critic(self, obs, option, action, next_obs, next_option, reward,
                    done, logger, step):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    next_obs = self.conv_input(next_obs, self.discrete_obs, self.obs_dim)

    option = self.conv_input(option, False, 1)
    next_option = self.conv_input(next_option, False, 1)

    with torch.no_grad():
      next_action, log_prob = self.actor.sample(next_obs, next_option)
      next_action = self.conv_input(next_action, self.actor.is_discrete(),
                                    self.action_dim)

      target_Q = self.critic_target(next_obs, next_option, next_action)
      target_V = target_Q - self.alpha.detach() * log_prob
      target_Q = reward + (1 - done) * self.gamma * target_V

    action = self.conv_input(action, self.actor.is_discrete(), self.action_dim)

    # get current Q estimates
    current_Q = self._critic(obs, option, action, both=True)
    if isinstance(current_Q, tuple):
      q1_loss = F.mse_loss(current_Q[0], target_Q)
      q2_loss = F.mse_loss(current_Q[1], target_Q)
      critic_loss = q1_loss + q2_loss
    else:
      critic_loss = F.mse_loss(current_Q, target_Q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if self.clip_grad_val:
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
    self.critic_optimizer.step()

    return {'loss/critic': critic_loss.item()}

  def update_actor_and_alpha(self, obs, option, logger, step):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    action, log_prob = self.actor.rsample(obs, option)
    actor_Q = self._critic(obs, option, action)

    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    if self.clip_grad_val:  # not zero or None
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_val)
    self.actor_optimizer.step()

    losses = {
        'loss/actor': actor_loss.item(),
        'actor_loss/target_entropy': self.target_entropy,
        'actor_loss/entropy': -log_prob.mean().item()
    }

    if self.learn_temp:
      self.log_alpha_optimizer.zero_grad()
      alpha_loss = (self.log_alpha *
                    (-log_prob - self.target_entropy).detach()).mean()

      alpha_loss.backward()
      self.log_alpha_optimizer.step()

      losses.update({
          'alpha_loss/loss': alpha_loss.item(),
          'alpha_loss/value': self.alpha.item(),
      })
    return losses

  # Save model parameters
  def save(self, path, suffix=""):
    actor_path = f"{path}{suffix}_actor"
    critic_path = f"{path}{suffix}_critic"

    # print('Saving models to {} and {}'.format(actor_path, critic_path))
    torch.save(self.actor.state_dict(), actor_path)
    torch.save(self._critic.state_dict(), critic_path)

  # Load model parameters
  def load(self, path):
    actor_path = f'{path}_actor'
    critic_path = f'{path}_critic'
    print('Loading models from {} and {}'.format(actor_path, critic_path))
    if actor_path is not None:
      self.actor.load_state_dict(
          torch.load(actor_path, map_location=self.device))
    if critic_path is not None:
      self._critic.load_state_dict(
          torch.load(critic_path, map_location=self.device))

  def log_probs(self, state, action):
    '''
    action should not be None. return shape is (len_demo, n_option)
    '''
    assert action is not None
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)
    action = self.conv_input(action, self.actor.is_discrete(), self.action_dim)

    log_probs = self.actor.log_prob_actions(state, action)

    return log_probs
