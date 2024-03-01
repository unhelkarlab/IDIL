from typing import Sequence
import warnings
import abc
import os
import numpy as np
import pickle
import models.mdp as mdp_lib
from models.utils.value_iteration import value_iteration


class PolicyInterface:
  '''
  to accomodate the case where a policy handles multiple actors,
  we consider actions are represented as a tuple
  '''

  __metaclass__ = abc.ABCMeta

  def __init__(self, mdp: mdp_lib.MDP) -> None:
    self.mdp = mdp

  @abc.abstractmethod
  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    '''
        returns the distribution of (joint) actions as the numpy array
    '''

    raise NotImplementedError

  def get_action(self, obstate_idx: int, latstate_idx: int) -> Sequence[int]:
    'most basic implementation - override if high performance is needed'
    np_action_dist = self.policy(obstate_idx, latstate_idx)
    aidx = np.random.choice(range(len(np_action_dist)), p=np_action_dist)
    return (aidx, )

  @abc.abstractmethod
  def conv_idx_to_action(self, tuple_aidx: Sequence[int]) -> Sequence:
    raise NotImplementedError

  @abc.abstractmethod
  def conv_action_to_idx(self, tuple_actions: Sequence) -> Sequence[int]:
    raise NotImplementedError

  @abc.abstractmethod
  def get_num_actions(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_num_latent_states(self):
    raise NotImplementedError

  @abc.abstractmethod
  def conv_idx_to_latent(self, latent_idx: int):
    raise NotImplementedError

  @abc.abstractmethod
  def conv_latent_to_idx(self, latent_state):
    raise NotImplementedError


class CachedPolicyInterface(PolicyInterface):

  def __init__(
      self,
      mdp: mdp_lib.LatentMDP,
      file_prefix: str,
      list_policy: list,
      temperature: float,
      queried_agent_indices: Sequence[int] = (0, )) -> None:
    '''
      queried_agent_indices: if the stored policy consists of joint actions and
                            if you want to query specific action factors
                            from the joint action, use this argument.
                            should be sorted in increasing order
    '''
    super().__init__(mdp)
    self.list_policy = list_policy
    self.file_prefix = file_prefix
    self.temperature = temperature
    self.queried_agent_indices = queried_agent_indices

    # for type hinting
    self.mdp = self.mdp  # type: mdp_lib.LatentMDP

  def prepare_policy(self):
    if (self.list_policy is None) or len(self.list_policy) == 0:
      for idx in range(self.get_num_latent_states()):
        str_q_val = self.file_prefix + "%d.pickle" % (idx, )
        np_q_value = None
        GAMMA = 0.95
        if self.file_prefix == "" or (not os.path.exists(str_q_val)):
          _, _, np_q_value = value_iteration(self.mdp.np_transition_model,
                                             self.mdp.np_reward_model[idx],
                                             discount_factor=GAMMA,
                                             max_iteration=500,
                                             epsilon=0.01)

          if self.file_prefix != "":
            dir_name = os.path.dirname(str_q_val)
            if not os.path.exists(dir_name):
              os.makedirs(dir_name)
            with open(str_q_val, "wb") as f:
              pickle.dump(np_q_value, f, pickle.HIGHEST_PROTOCOL)
        else:
          with open(str_q_val, "rb") as f:
            np_q_value = pickle.load(f)

          warnings.warn(
              "The Q-values for the policy has been loaded from a file ({}). "
              "If any related implementation is changed, "
              "be sure to delete the saved file and regenerate it.".format(
                  "prefix: " + os.path.basename(self.file_prefix)),
              stacklevel=2)

        self.list_policy.append(
            mdp_lib.softmax_policy_from_q_value(np_q_value, self.temperature))

  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    '''
    return: 1-D distribution of the joint action
    NOTE: can be slow if the joint action consists of multiple factors
    '''
    self.prepare_policy()

    # if the entire joint actions are queried, marginalization is not needed
    if len(self.mdp.list_num_actions) == len(self.queried_agent_indices):
      return self.list_policy[latstate_idx][obstate_idx, :]

    np_action_dist = np.reshape(self.list_policy[latstate_idx][obstate_idx, :],
                                self.mdp.list_num_actions)

    # marginalize out residual actions
    axis2sum = [
        idx for idx in range(self.mdp.num_action_factors)
        if idx not in self.queried_agent_indices
    ]
    np_action_dist = np_action_dist.sum(
        axis=tuple(axis2sum))  # type: np.ndarray
    return np_action_dist.ravel()

  def get_action(self, obstate_idx: int, latstate_idx: int):
    self.prepare_policy()

    joint_aidx = np.random.choice(
        range(self.mdp.num_actions),
        size=1,
        replace=False,
        p=self.list_policy[latstate_idx][obstate_idx, :])[0]
    vector_indv_aidx = self.mdp.conv_idx_to_action(joint_aidx)
    return vector_indv_aidx[list(self.queried_agent_indices)]

  def conv_idx_to_action(self, tuple_aidx: Sequence[int]):
    list_actions = []
    for idx, fidx in enumerate(self.queried_agent_indices):
      list_actions.append(
          self.mdp.dict_factored_actionspace[fidx].idx_to_action[
              tuple_aidx[idx]])

    return list_actions

  def conv_action_to_idx(self, tuple_actions: Sequence) -> Sequence[int]:
    list_aidx = []
    for idx, fidx in enumerate(self.queried_agent_indices):
      list_aidx.append(self.mdp.dict_factored_actionspace[fidx].action_to_idx[
          tuple_actions[idx]])

    return list_aidx

  def get_num_actions(self):
    if len(self.queried_agent_indices) == 1:
      return self.mdp.list_num_actions[self.queried_agent_indices[0]]
    else:
      list_num_actions = []
      for fidx in self.queried_agent_indices:
        list_num_actions.append(self.mdp.list_num_actions[fidx])

      return tuple(list_num_actions)

  def get_num_latent_states(self):
    return self.mdp.num_latents

  def conv_idx_to_latent(self, latent_idx: int):
    return self.mdp.latent_space.idx_to_state[latent_idx]

  def conv_latent_to_idx(self, latent_state):
    return self.mdp.latent_space.state_to_idx[latent_state]
