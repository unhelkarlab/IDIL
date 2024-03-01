import abc
from typing import Optional, Sequence
import numpy as np
from models.policy import PolicyInterface


class AgentModel:
  __metaclass__ = abc.ABCMeta

  def __init__(self, policy_model: Optional[PolicyInterface] = None) -> None:
    self.policy_model = policy_model
    self.current_latent = -1  # type: int

  def set_policy(self, policy_model):
    self.policy_model = policy_model

  def is_current_latent_valid(self):
    return self.current_latent >= 0

  @abc.abstractmethod
  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    'return: 1-D array'
    raise NotImplementedError

  @abc.abstractmethod
  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    'return: 1-D array'
    raise NotImplementedError

  def get_reference_mdp(self):
    return self.policy_model.mdp

  def sample_initial_mental_state(self, obstate_idx: int) -> int:
    np_init_dist = self.initial_mental_distribution(obstate_idx)
    return np.random.choice(range(len(np_init_dist)), p=np_init_dist)

  def sample_next_mental_state(self, latstate_idx: int, obstate_idx: int,
                               tuple_action_idx: Sequence[int],
                               obstate_next_idx: int) -> int:
    np_next_latent_dist = self.transition_mental_state(latstate_idx,
                                                       obstate_idx,
                                                       tuple_action_idx,
                                                       obstate_next_idx)
    return np.random.choice(range(len(np_next_latent_dist)),
                            p=np_next_latent_dist)

  def set_init_mental_state_idx(self,
                                obstate_idx: int,
                                init_latent: Optional[int] = None):
    '''if you want to set latent state directly, set init_latent.
    In this case, any value set to obstate_idx will be ignored.
    Returns the latent just set for external analysis'''
    if init_latent is None:
      self.current_latent = self.sample_initial_mental_state(obstate_idx)
    else:
      self.current_latent = init_latent

    return self.current_latent

  def update_mental_state_idx(self, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int):
    'update latent state internally and return it for external analysis'
    self.current_latent = self.sample_next_mental_state(self.current_latent,
                                                        obstate_idx,
                                                        tuple_action_idx,
                                                        obstate_next_idx)
    return self.current_latent

  def get_action_idx(self, obstate_idx: int) -> Sequence[int]:
    return self.policy_model.get_action(obstate_idx, self.current_latent)
