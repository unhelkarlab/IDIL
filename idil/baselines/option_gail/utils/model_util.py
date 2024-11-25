import math
import torch
from torch import nn
from typing import Type
from idil.baselines.IQLearn.utils.utils import one_hot
import numpy as np


def conv_nn_input(batch_input, is_onehot_needed, dim, device):
  if is_onehot_needed:
    if not isinstance(batch_input, torch.Tensor):
      batch_input = torch.tensor(batch_input,
                                 dtype=torch.float).reshape(-1).to(device)
    else:
      batch_input = batch_input.reshape(-1)
    batch_input = one_hot(batch_input, dim)
  else:
    if not isinstance(batch_input, torch.Tensor):
      batch_input = torch.tensor(np.array(batch_input).reshape(-1, dim),
                                 dtype=torch.float).to(device)

  return batch_input


def init_layer(module, gain=math.sqrt(2)):
  with torch.no_grad():
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
  return module


def make_module(in_size,
                out_size,
                hidden,
                activation: Type[nn.Module] = nn.ReLU):
  n_in = in_size
  l_hidden = []
  for h in hidden:
    l_hidden.append(init_layer(torch.nn.Linear(n_in, h)))
    l_hidden.append(activation())
    n_in = h
  l_hidden.append(init_layer(torch.nn.Linear(n_in, out_size), gain=0.1))
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
