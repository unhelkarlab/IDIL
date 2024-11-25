from typing import Type
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from .sac_models import DiscreteActor
from ..utils.utils import soft_update, one_hot
from omegaconf import DictConfig


class SAC_Discrete(object):

  def __init__(self, config: DictConfig, obs_dim, action_dim, discrete_obs,
               critic_base: Type[nn.Module], actor: DiscreteActor):
    self.gamma = config.gamma
    self.batch_size = config.mini_batch_size
    self.discrete_obs = discrete_obs
    self.obs_dim = obs_dim
    self.action_dim = action_dim

    self.device = torch.device(config.device)

    self.clip_grad_val = config.clip_grad_val
    self.critic_tau = 0.005
    self.learn_temp = False
    self.actor_update_frequency = 1
    self.critic_target_update_frequency = 1
    use_tanh = False
    self.init_temp = config.init_temp

    self._critic = critic_base(obs_dim, action_dim, config.hidden_critic,
                               self.gamma, use_tanh).to(self.device)

    self.critic_target = critic_base(obs_dim, action_dim, config.hidden_critic,
                                     self.gamma, use_tanh).to(self.device)

    self.critic_target.load_state_dict(self._critic.state_dict())

    self.actor = actor.to(self.device)

    self.log_alpha = torch.tensor(np.log(self.init_temp)).to(self.device)
    self.log_alpha.requires_grad = True
    # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
    self.target_entropy = -action_dim

    self.USE_EXACT_POLICY_FOR_V = False

    # optimizers
    actor_betas = critic_betas = alpha_betas = [0.9, 0.999]
    self.actor_optimizer = Adam(self.actor.parameters(),
                                lr=config.optimizer_lr_policy,
                                betas=actor_betas)
    self.critic_optimizer = Adam(self._critic.parameters(),
                                 lr=config.optimizer_lr_critic,
                                 betas=critic_betas)
    self.log_alpha_optimizer = Adam([self.log_alpha],
                                    lr=config.optimizer_lr_alpha,
                                    betas=alpha_betas)
    self.train()
    self.critic_target.train()

  def train(self, training=True):
    self.training = training
    self.actor.train(training)
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

  def choose_action(self, state, sample=False):
    # --- convert state
    if self.discrete_obs:
      state = torch.FloatTensor([state]).to(self.device)
      state = one_hot(state, self.obs_dim)
      state = state.view(-1, self.obs_dim)
    # ------
    else:
      state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

    with torch.no_grad():
      if sample:
        action, _ = self.actor.sample(state)
      else:
        action = self.actor.exploit(state)

    return action.detach().cpu().numpy()[0]

  def critic(self, obs, action, both=False):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    q = self._critic(obs, both)
    if isinstance(q, tuple) and both:
      q1, q2 = q
      critic1 = q1.gather(1, action.long())
      critic2 = q2.gather(1, action.long())
      return critic1, critic2

    return q.gather(1, action.long())

  def getV(self, obs):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    action_probs, log_action_probs = self.actor.action_probs(obs)

    current_Q = self._critic(obs)

    if self.USE_EXACT_POLICY_FOR_V:
      current_V = self.alpha * torch.logsumexp(
          current_Q / self.alpha, dim=1, keepdim=True)
    else:
      current_V = (action_probs *
                   (current_Q - self.alpha * log_action_probs)).sum(
                       dim=1, keepdim=True)
    return current_V

  def get_targetV(self, obs):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    action_probs, log_action_probs = self.actor.action_probs(obs)

    target_Q = self.critic_target(obs)

    if self.USE_EXACT_POLICY_FOR_V:
      target_V = self.alpha * torch.logsumexp(
          target_Q / self.alpha, dim=1, keepdim=True)
    else:
      target_V = (action_probs *
                  (target_Q - self.alpha * log_action_probs)).sum(dim=1,
                                                                  keepdim=True)
    return target_V

  def update(self, replay_buffer, logger, step):
    obs, next_obs, action, reward, done = replay_buffer.get_samples(
        self.batch_size, self.device)

    losses = self.update_critic(obs, action, reward, next_obs, done, logger,
                                step)

    if step % self.actor_update_frequency == 0:
      actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
      losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
      soft_update(self._critic, self.critic_target, self.critic_tau)

    return losses

  def update_critic(self, obs, action, reward, next_obs, done, logger, step):

    with torch.no_grad():
      target_V = self.get_targetV(next_obs)
      target_Q = reward + (1 - done) * self.gamma * target_V

    # get current Q estimates
    current_Q = self.critic(obs, action, both=True)
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
    if self.clip_grad_val:  # not zero or None
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
    self.critic_optimizer.step()

    return {'loss/critic': critic_loss.item()}

  def update_actor_and_alpha(self, obs, logger, step):

    # --- convert state
    if self.discrete_obs:
      obs = one_hot(obs, self.obs_dim)
    # ------

    with torch.no_grad():
      actor_Q = self._critic(obs)

    action_probs, log_action_probs = self.actor.action_probs(obs)

    entropies = -(action_probs * log_action_probs).sum(dim=1, keepdim=True)
    avg_Q = (action_probs * actor_Q).sum(dim=1, keepdim=True)

    actor_loss = (-avg_Q - self.alpha.detach() * entropies).mean()

    # logger.log('train/actor_loss', actor_loss, step)
    # logger.log('train/target_entropy', self.target_entropy, step)
    # logger.log('train/actor_entropy', -entropies.mean(), step)

    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    if self.clip_grad_val:  # not zero or None
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_val)
    self.actor_optimizer.step()

    losses = {
        'loss/actor': actor_loss.item(),
        'actor_loss/target_entropy': self.target_entropy,
        'actor_loss/entropy': -entropies.mean().item()
    }

    if self.learn_temp:
      self.log_alpha_optimizer.zero_grad()
      alpha_loss = (self.log_alpha *
                    (entropies - self.target_entropy).detach()).mean()
      # logger.log('train/alpha_loss', alpha_loss, step)
      # logger.log('train/alpha_value', self.alpha, step)

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
