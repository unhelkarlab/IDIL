import torch
from ..utils.model_util import make_module, make_module_list, make_activation, conv_nn_input
from omegaconf import DictConfig

# This file should be included by option_ppo.py and never be used otherwise


class Critic(torch.nn.Module):

  def __init__(self, config: DictConfig, dim_s):
    super(Critic, self).__init__()
    self.dim_s = dim_s
    self.device = torch.device(config.device)

    activation = make_activation(config.activation)
    n_hidden_v = config.hidden_critic

    self.value = make_module(self.dim_s, 1, n_hidden_v, activation)

    self.to(self.device)

  def get_value(self, s):
    return self.value(s)

  def get_param(self):
    return list(self.parameters())


class OptionCritic(torch.nn.Module):

  def __init__(self, config, dim_s, dim_c,
               discrete_s=False):
    super(OptionCritic, self).__init__()
    self.dim_s = dim_s
    self.dim_c = dim_c
    self.discrete_s = discrete_s
    self.device = torch.device(config.device)
    self.is_shared = config.shared_critic

    activation = make_activation(config.activation)
    n_hidden_v = config.hidden_critic

    if self.is_shared:
      self.value = make_module(self.dim_s, self.dim_c, n_hidden_v, activation)
    else:
      self.value = make_module_list(self.dim_s, 1, n_hidden_v, self.dim_c,
                                    activation)

    self.to(self.device)

  def get_value(self, s, c=None):
    s = conv_nn_input(s, self.discrete_s, self.dim_s, self.device)
    # c could be None for directly output value on each c
    if self.is_shared:
      vs = self.value(s)
    else:
      vs = torch.cat([v(s) for v in self.value], dim=-1)

    if c is None:
      return vs
    else:
      return vs.gather(dim=-1, index=c)

  def get_param(self):
    return list(self.parameters())
