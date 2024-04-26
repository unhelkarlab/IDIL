import os
from typing import Type
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from idil_algs.baselines.IQLearn.utils.utils import one_hot
from .option_abstract import AbstractPolicyLeaner
from omegaconf import DictConfig

USE_TARGET = False


class OptionSoftQ(AbstractPolicyLeaner):

  def __init__(self, config: DictConfig, num_inputs, action_dim, option_dim,
               discrete_obs, q_net_base: Type[nn.Module]):
    super().__init__(config)
    self.critic_tau = 0.1
    self.init_temp = config.init_temp
    use_tanh = False

    self.discrete_obs = discrete_obs
    self.obs_dim = num_inputs

    self.critic_target_update_frequency = 4
    self.log_alpha = torch.tensor(np.log(self.init_temp)).to(self.device)

    self.q_net = q_net_base(num_inputs, action_dim, option_dim,
                            config.hidden_critic, config.activation, self.gamma,
                            use_tanh).to(self.device)
    if USE_TARGET:
      self.target_net = q_net_base(num_inputs, action_dim, option_dim,
                                   config.hidden_critic, config.activation,
                                   self.gamma, use_tanh).to(self.device)

      self.target_net.load_state_dict(self.q_net.state_dict())

    # optimizers
    self.optimizer_lr_critic = config.optimizer_lr_critic
    self.reset_optimizers()

    self.train()
    if USE_TARGET:
      self.target_net.train()

  def train(self, training=True):
    self.training = training
    self.q_net.train(training)

  def reset_optimizers(self):
    critic_betas = [0.9, 0.999]
    self.critic_optimizer = Adam(self.q_net.parameters(),
                                 lr=self.optimizer_lr_critic,
                                 betas=critic_betas)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  def critic_net(self):
    return self.q_net

  @property
  def critic_target_net(self):
    return self.target_net

  def choose_action(self, obs, option, sample=False):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    with torch.no_grad():
      q = self.q_net(obs, option)
      dist = F.softmax(q / self.alpha, dim=-1)
      dist = Categorical(dist)
      action = dist.sample()

    return action.detach().cpu().numpy()[0]

  def critic(self, obs, option, action, both=False):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)
    action = self.conv_input(action, False, 1)

    q = self.q_net(obs, option, both)
    if isinstance(q, tuple) and both:
      q1, q2 = q
      critic1 = q1.gather(1, action.long())
      critic2 = q2.gather(1, action.long())
      return critic1, critic2

    return q.gather(1, action.long())

  def getV(self, obs, option):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    q = self.q_net(obs, option)
    v = self.alpha * torch.logsumexp(q / self.alpha, dim=-1, keepdim=True)
    return v

  def get_targetV(self, obs, option):

    obs = self.conv_input(obs, self.discrete_obs, self.obs_dim)
    option = self.conv_input(option, False, 1)

    q = self.target_net(obs, option)
    target_v = self.alpha * torch.logsumexp(
        q / self.alpha, dim=-1, keepdim=True)
    return target_v

  def update(self, obs, option, action, next_obs, next_option, reward, done,
             logger, step):

    losses = self.update_critic(obs, option, action, next_obs, next_option,
                                reward, done, logger, step)

    if step % self.critic_target_update_frequency == 0:
      self.target_net.load_state_dict(self.q_net.state_dict())

    return losses

  def update_critic(self, obs, option, action, next_obs, next_option, reward,
                    done, logger, step):

    with torch.no_grad():
      next_v = self.get_targetV(next_obs, next_option)
      y = reward + (1 - done) * self.gamma * next_v

    critic_loss = F.mse_loss(self.critic(obs, option, action), y)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    return {'loss/critic': critic_loss.item()}

  # Save model parameters
  def save(self, path, suffix=""):
    critic_path = f"{path}{suffix}"
    # print('Saving models to {} and {}'.format(actor_path, critic_path))
    torch.save(self.q_net.state_dict(), critic_path)

  # Load model parameters
  def load(self, path):
    critic_path = f'{path}'
    print('Loading models from {}'.format(critic_path))
    self.q_net.load_state_dict(torch.load(critic_path,
                                          map_location=self.device))

  def log_probs(self, state, action):
    '''
    if action is not None: return shape is (len_demo, n_option)
    if action is None: return shape is (len_demo, n_option, n_action)
    '''
    state = self.conv_input(state, self.discrete_obs, self.obs_dim)

    qout = self.q_net(state, None)
    log_probs = F.log_softmax(qout / self.alpha, dim=-1)
    if action is not None:
      action = self.conv_input(action, False, 1)
      action_idx = action.long().view(-1, 1, 1).expand(-1, log_probs.shape[-2],
                                                       1)
      log_probs = log_probs.gather(dim=-1, index=action_idx).squeeze(-1)

    return log_probs
