from typing import Type
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from idil.baselines.IQLearn.agent.sac_models import (SquashedNormal,
                                                     GumbelSoftmax)


def weight_init(m):
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)


def make_module(in_size,
                out_size,
                hidden,
                activation: Type[nn.Module] = nn.ReLU):
  n_in = in_size
  l_hidden = []
  for h in hidden:
    l_hidden.append(torch.nn.Linear(n_in, h))
    l_hidden.append(torch.nn.ReLU())
    n_in = h
  l_hidden.append(torch.nn.Linear(n_in, out_size))
  return torch.nn.Sequential(*l_hidden)


def make_module_list(in_size,
                     out_size,
                     hidden,
                     n_net,
                     activation: Type[nn.Module] = nn.ReLU):
  return nn.ModuleList([
      make_module(in_size, out_size, hidden, activation) for _ in range(n_net)
  ])


def make_activation(act_name):
  return (torch.nn.ReLU if act_name == "relu" else torch.nn.Tanh
          if act_name == "tanh" else torch.nn.Sigmoid if act_name == "sigmoid"
          else torch.nn.Softplus if act_name == "softplus" else None)


# #############################################################################
# SoftQ models
class OptionSoftQNetwork(nn.Module):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__()
    self.use_tanh = use_tanh
    self.gamma = gamma


class SimpleOptionQNetwork(OptionSoftQNetwork):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    # Q1 architecture
    self.Q1 = make_module_list(obs_dim, action_dim, list_hidden_dims,
                               option_dim, activation)

    self.apply(weight_init)

  def forward(self, state, option, *args):
    '''
    if option is None, the shape of return is (n_batch, option_dim, action_dim)
    otherwise, the shape of return is (n_batch, action_dim)
    '''
    out = torch.stack([m(state) for m in self.Q1], dim=-2)

    if option is not None:
      option_idx = option.long().view(-1, 1, 1).expand(-1, 1, out.shape[-1])
      out = out.gather(dim=-2, index=option_idx).squeeze(-2)

    if self.use_tanh:
      out = torch.tanh(out) * 1 / (1 - self.gamma)

    return out


class DoubleOptionQNetwork(OptionSoftQNetwork):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    self.net1 = make_module_list(obs_dim, action_dim, list_hidden_dims,
                                 option_dim, activation)
    self.net2 = make_module_list(obs_dim, action_dim, list_hidden_dims,
                                 option_dim, activation)

    self.apply(weight_init)

  def forward(self, state, option, both=False, *args):
    '''
    if option is None, the shape of return is (n_batch, option_dim, action_dim)
    otherwise, the shape of return is (n_batch, action_dim)
    '''
    q1 = torch.stack([m(state) for m in self.net1], dim=-2)
    q2 = torch.stack([m(state) for m in self.net2], dim=-2)

    if option is not None:
      option_idx = option.long().view(-1, 1, 1).expand(-1, 1, q1.shape[-1])
      q1 = q1.gather(dim=-2, index=option_idx).squeeze(-2)
      q2 = q2.gather(dim=-2, index=option_idx).squeeze(-2)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.minimum(q1, q2)


# #############################################################################
# SACQCritic models


class SACOptionQCritic(nn.Module):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__()
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.use_tanh = use_tanh
    self.gamma = gamma


class DoubleOptionQCritic(SACOptionQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    self.Q1 = make_module_list(obs_dim + action_dim, 1, list_hidden_dims,
                               option_dim, activation)
    self.Q2 = make_module_list(obs_dim + action_dim, 1, list_hidden_dims,
                               option_dim, activation)

    self.apply(weight_init)

  def forward(self, obs, option, action, both=False, *args):
    '''
    if option is None, the shape of return is (n_batch, option_dim, 1)
    otherwise, the shape of return is (n_batch, 1)
    '''
    obs_action = torch.cat([obs, action], dim=-1)
    q1 = torch.cat([m(obs_action) for m in self.Q1], dim=-1)
    q2 = torch.cat([m(obs_action) for m in self.Q2], dim=-1)

    if option is not None:
      q1 = q1.gather(dim=-1, index=option.long())
      q2 = q2.gather(dim=-1, index=option.long())

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.min(q1, q2)


class SingleOptionQCritic(SACOptionQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, option_dim, list_hidden_dims,
                     activation, gamma, use_tanh)

    activation = make_activation(activation)
    self.Q1 = make_module_list(obs_dim + action_dim, 1, list_hidden_dims,
                               option_dim, activation)

    self.apply(weight_init)

  def forward(self, obs, option, action, both=False, *args):
    '''
    if option is None, the shape of return is (n_batch, option_dim, 1)
    otherwise, the shape of return is (n_batch, 1)
    '''
    obs_action = torch.cat([obs, action], dim=-1)
    q1 = torch.cat([m(obs_action) for m in self.Q1], dim=-1)

    if option is not None:
      q1 = q1.gather(dim=-1, index=option.long())

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)

    return q1


# #############################################################################
# Actor models


class AbstractOptionActor(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, obs, option):
    raise NotImplementedError

  def rsample(self, obs, option):
    raise NotImplementedError

  def sample(self, obs, option):
    raise NotImplementedError

  def exploit(self, obs, option):
    raise NotImplementedError

  def is_discrete(self):
    raise NotImplementedError

  def log_prob_actions(self, obs, action):
    raise NotImplementedError


class DiagGaussianOptionActor(AbstractOptionActor):
  """torch.distributions implementation of an diagonal Gaussian policy."""

  def __init__(self,
               obs_dim,
               action_dim,
               option_dim,
               list_hidden_dims,
               activation,
               log_std_bounds,
               bounded=True,
               use_nn_logstd=False,
               clamp_action_logstd=False):
    super().__init__()
    self.use_nn_logstd = use_nn_logstd
    self.clamp_action_logstd = clamp_action_logstd

    output_dim = action_dim
    if self.use_nn_logstd:
      output_dim = 2 * action_dim
    else:
      self.action_logstd = nn.Parameter(
          torch.empty(1, action_dim, dtype=torch.float32).fill_(0.))

    activation = make_activation(activation)

    self.trunk = make_module_list(obs_dim, output_dim, list_hidden_dims,
                                  option_dim, activation)

    self.log_std_bounds = log_std_bounds
    self.bounded = bounded

    self.apply(weight_init)

  def forward(self, obs, option):
    '''
    if option is None, the shape of return is (n_batch, option_dim, action_dim)
    otherwise, the shape of return is (n_batch, action_dim)
    '''
    out = torch.stack([m(obs) for m in self.trunk], dim=-2)

    if option is not None:
      option_idx = option.long().view(-1, 1, 1).expand(-1, 1, out.shape[-1])
      out = out.gather(dim=-2, index=option_idx).squeeze(-2)

    if self.use_nn_logstd:
      mu, log_std = out.chunk(2, dim=-1)
    else:
      mu = out
      log_std = self.action_logstd.expand_as(mu)

    # clamp logstd
    if self.clamp_action_logstd:
      log_std = log_std.clamp(self.log_std_bounds[0], self.log_std_bounds[1])
    else:
      # constrain log_std inside [log_std_min, log_std_max]
      log_std = torch.tanh(log_std)
      log_std_min, log_std_max = self.log_std_bounds
      log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    std = log_std.exp()

    if self.bounded:
      dist = SquashedNormal(mu, std)
    else:
      mu = mu.clamp(-10, 10)
      dist = Normal(mu, std)

    return dist

  def rsample(self, obs, option):
    dist = self.forward(obs, option)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def sample(self, obs, option):
    return self.rsample(obs, option)

  def exploit(self, obs, option):
    return self.forward(obs, option).mean

  def is_discrete(self):
    return False

  def log_prob_actions(self, obs, action):
    if self.bounded:
      EPS = 1.e-7
      action = action.clip(-1.0 + EPS, 1.0 - EPS)

    dist = self.forward(obs, None)
    action = action.view(-1, 1, dist.loc.shape[-1])
    return dist.log_prob(action).sum(-1, keepdim=True)
