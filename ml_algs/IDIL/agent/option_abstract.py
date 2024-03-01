import abc
import torch
import numpy as np
from ml_algs.baselines.IQLearn.utils.utils import one_hot
from omegaconf import DictConfig


class AbstractPolicyLeaner(abc.ABC):

  def __init__(self, config: DictConfig):
    self.gamma = config.gamma
    self.device = torch.device(config.device)
    self.actor = None
    self.clip_grad_val = config.clip_grad_val
    self.num_critic_update = config.num_critic_update
    self.num_actor_update = config.num_actor_update

  def conv_input(self, batch_input, is_onehot_needed, dimension):
    if is_onehot_needed:
      if not isinstance(batch_input, torch.Tensor):
        batch_input = torch.tensor(
            batch_input, dtype=torch.float).reshape(-1).to(self.device)
      else:
        batch_input = batch_input.reshape(-1)
      batch_input = one_hot(batch_input, dimension)
    else:
      if not isinstance(batch_input, torch.Tensor):
        batch_input = torch.tensor(np.array(batch_input).reshape(-1, dimension),
                                   dtype=torch.float).to(self.device)

    return batch_input

  @abc.abstractmethod
  def reset_optimizers(self, config: DictConfig):
    pass

  @abc.abstractmethod
  def train(self, training=True):
    pass

  @property
  @abc.abstractmethod
  def alpha(self):
    pass

  @property
  @abc.abstractmethod
  def critic_net(self):
    pass

  @property
  @abc.abstractmethod
  def critic_target_net(self):
    pass

  @abc.abstractmethod
  def choose_action(self, state, option, sample=False):
    pass

  @abc.abstractmethod
  def critic(self, obs, option, action, both=False):
    pass

  @abc.abstractmethod
  def getV(self, obs, option):
    pass

  @abc.abstractmethod
  def get_targetV(self, obs, option):
    pass

  @abc.abstractmethod
  def update(self, obs, option, action, next_obs, next_option, reward, done,
             logger, step):
    pass

  @abc.abstractmethod
  def update_critic(self, obs, option, action, next_obs, next_option, reward,
                    done, logger, step):
    pass

  @abc.abstractmethod
  def save(self, path, suffix=""):
    pass

  @abc.abstractmethod
  def load(self, path):
    pass

  @abc.abstractmethod
  def log_probs(self, state, action):
    pass
