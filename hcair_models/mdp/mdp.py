"""Defines an abstract MDP class using numpy."""

import abc
from typing import Optional, Tuple, Union

import logging
import numpy as np
import numpy.testing as npt
import scipy.special as sc
from tqdm import tqdm
import sparse

from hcair_models.utils.mdp_utils import StateSpace, ActionSpace


class MDP:
  """Abstract MDP class."""
  # Python 2 style but Python3 is also compatible with this.
  __metaclass__ = abc.ABCMeta

  def __init__(self, fast_cache_mode: bool = False, use_sparse: bool = False):
    """Initializes MDP class.

      Args:
        fast_cache_mode: Enables fast caching by avoiding assertions. Note
          this should be true only after verifying the implementation is
          correct.
    """
    self.fast_cache_mode = fast_cache_mode
    self.use_sparse = use_sparse

    # Define state space.
    self.init_statespace()
    self.init_statespace_helper_vars()

    # Define action space.
    self.init_actionspace()
    self.init_actionspace_helper_vars()

    # Define empty transition and reward numpy ndarrays.
    # We do not compute them here, as the computations might be costly.
    # Users can compute them by calling the respective properties.
    self._np_transition_model = None
    self._np_reward_model = None

  @abc.abstractmethod
  def init_statespace(self):
    """Defines MDP state space.

    The MDP class allows definition of factored state spaces.
    The method should define a dictionary mapping state feature indices
    to its state space. The source provides an example implementation.
    The joint state space will be automatically constructed using the
    helper method init_statespace_helper_vars.

    If needed, dummy states can be defined with self.dummy_states variable.
    To disable dummy states, set self.dummy_states = None
    """
    self.dict_factored_statespace = {}
    s0_space = StateSpace()
    s1_space = StateSpace()
    self.dict_factored_statespace = {0: s0_space, 1: s1_space}

    self.dummy_states = None
    self.dummy_states = StateSpace()

  def init_statespace_helper_vars(self):
    """Creates helper variables for the state space."""

    # Retrieve number of states and state factors.
    self.num_state_factors = len(self.dict_factored_statespace)
    self.list_num_states = []
    for idx in range(self.num_state_factors):
      self.list_num_states.append(
          self.dict_factored_statespace.get(idx).num_states)

    self.num_actual_states = np.prod(self.list_num_states)
    self.num_dummy_states = (0 if self.dummy_states is None else
                             self.dummy_states.num_states)
    self.num_states = self.num_actual_states + self.num_dummy_states
    logging.debug("num_states= %d" % (self.num_states, ))

    # Create mapping from state to state index.
    # Mapping takes state value as inputs and outputs a scalar state index.
    np_list_idx = np.arange(self.num_actual_states, dtype=np.int32)
    self.np_state_to_idx = np_list_idx.reshape(self.list_num_states)

    # Create mapping from state index to state.
    # Mapping takes state index as input and outputs a factored state.
    np_idx_to_state = np.zeros((self.num_actual_states, self.num_state_factors),
                               dtype=np.int32)
    for state, idx in np.ndenumerate(self.np_state_to_idx):
      np_idx_to_state[idx] = state
    self.np_idx_to_state = np_idx_to_state

  def is_dummy_state(self, idx: int):
    """check if it's dummy state or not"""
    return idx >= self.num_actual_states and idx < self.num_states

  def conv_state_to_idx(self, tuple_state: Tuple[int, ...]):
    return self.np_state_to_idx[tuple_state]

  def conv_idx_to_state(self, idx: int):
    assert idx < self.num_actual_states, "state index is out of range"
    return self.np_idx_to_state[idx]

  def conv_idx_to_dummy_state(self, idx: int):
    assert (idx >= self.num_actual_states
            and idx < self.num_states), ("dummy state index is out of range")
    return self.dummy_states.idx_to_state[idx - self.num_actual_states]

  def conv_dummy_state_to_idx(self, dummy_state):
    return self.dummy_states.state_to_idx[dummy_state] + self.num_actual_states

  @abc.abstractmethod
  def conv_sim_states_to_mdp_sidx(self, tup_states):
    '''
    intentionally did not specify the type or shape of the input argument
    to provide child classes some freedom.

    Example:

      list_indv_sidx = []
      for idx, state in enumerate(tup_states):
        indv_sidx = self.dict_factored_statespace[idx].state_to_idx[state]
        list_indv_sidx.append(indv_sidx)

      return self.conv_state_to_idx(tuple(list_indv_sidx))
    '''
    raise NotImplementedError

  @abc.abstractmethod
  def conv_mdp_sidx_to_sim_states(self, state_idx):
    raise NotImplementedError

  @abc.abstractmethod
  def conv_mdp_aidx_to_sim_actions(self, action_idx):
    raise NotImplementedError

  @abc.abstractmethod
  def conv_sim_actions_to_mdp_aidx(self, tuple_actions):
    raise NotImplementedError

  @abc.abstractmethod
  def init_actionspace(self):
    """Defines MDP action space.

    The MDP class allows definition of factored action spaces.
    The method should define a dictionary mapping action feature indices
    to its action space. The source provides an example implementation.
    The joint action space will be automatically constructed using the
    helper method init_actionspace_helper_vars.
    """
    self.dict_factored_actionspace = {}
    a0_space = ActionSpace()
    a1_space = ActionSpace()
    self.dict_factored_actionspace = {0: a0_space, 1: a1_space}

  def init_actionspace_helper_vars(self):
    """Creates helper variables for the action space."""

    # Retrieve number of actions and action factors.
    self.num_action_factors = len(self.dict_factored_actionspace)
    self.list_num_actions = []
    for idx in range(self.num_action_factors):
      self.list_num_actions.append(
          self.dict_factored_actionspace.get(idx).num_actions)
    self.num_actions = np.prod(self.list_num_actions)
    logging.debug("num_actions= %d" % (self.num_actions, ))

    # Create mapping from action to action index.
    # Mapping takes action value as inputs and outputs a scalar action index.
    np_list_idx = np.arange(self.num_actions, dtype=np.int32)
    self.np_action_to_idx = np_list_idx.reshape(self.list_num_actions)

    # Create mapping from action index to action.
    # Mapping takes action index as input and outputs a factored action.
    np_idx_to_action = np.zeros((self.num_actions, self.num_action_factors),
                                dtype=np.int32)
    for action, idx in np.ndenumerate(self.np_action_to_idx):
      np_idx_to_action[idx] = action
    self.np_idx_to_action = np_idx_to_action

  def conv_action_to_idx(self, tuple_action: Tuple[int, ...]):
    return self.np_action_to_idx[tuple_action]

  def conv_idx_to_action(self, idx: int):
    return self.np_idx_to_action[idx]

  @abc.abstractmethod
  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    """Defines MDP transition function.

      Args:
        state_idx: Index of an MDP state.
        action_idx: Index of an MDP action.

      Returns:
        A numpy array with two columns and at least one row.
        The first column corresponds to the probability for the next state.
        The second column corresponds to the index of the next state.
    """
    raise NotImplementedError
    return np_next_p_state_idx  # noqa: F821

  @property
  def np_transition_model(self) -> np.ndarray:
    """Returns transition model as a np ndarray."""

    # If already computed, return the computed value.
    # This model does not change after the MDP is defined.
    if self._np_transition_model is not None:
      return self._np_transition_model

    if not self.use_sparse:
      logging.debug("Transition model type: numpy array")
      # Else: Compute using the transition model.
      self._np_transition_model = np.zeros(
          (self.num_states, self.num_actions, self.num_states),
          dtype=np.float32)
      for state in tqdm(range(self.num_states)):
        # for illegal actions and termial states,
        # the transition should remain at the same state
        self._np_transition_model[state, :, state] = np.ones(
            (self.num_actions, ))
        for action in self.legal_actions(state):
          self._np_transition_model[state, action, state] = 0
          np_next_p_state_idx = self.transition_model(state, action)
          if self.fast_cache_mode:
            self._np_transition_model[state, action,
                                      tuple(np_next_p_state_idx[:,
                                                                1].astype(int)
                                            )] = (np_next_p_state_idx[:, 0])
          else:
            for (next_p, next_state) in np_next_p_state_idx:
              next_state = int(next_state)
              self._np_transition_model[state, action, next_state] = next_p
            npt.assert_almost_equal(
                actual=self._np_transition_model[state, action].sum(),
                desired=1,
                decimal=7,
                err_msg="Transition probabilities do not sum to one.",
            )
    else:
      logging.debug("Transition model type: Sparse")
      # coord_2_data = {
      #     (self.num_states - 1, self.num_actions - 1, self.num_states - 1): 0
      # }
      coord_2_data = {}
      for state in tqdm(range(self.num_states)):
        # for illegal actions and termial states,
        # the transition should remain at the same state
        for action in range(self.num_actions):
          coord_2_data[(state, action, state)] = 1

        for action in self.legal_actions(state):
          coord_2_data[(state, action, state)] = 0
          np_next_p_state_idx = self.transition_model(state, action)

          for (next_p, next_state) in np_next_p_state_idx:
            next_state = int(next_state)
            coord_2_data[(state, action, next_state)] = np.float32(next_p)

      self._np_transition_model = sparse.COO.from_iter(coord_2_data,
                                                       shape=(self.num_states,
                                                              self.num_actions,
                                                              self.num_states),
                                                       dtype=np.float32)

    return self._np_transition_model

  def transition(self, state_idx: int, action_idx: int) -> int:
    """Samples next state using the MDP transition function / model.

    Args:
      state_idx: Index of an MDP state.
      action_idx: Index of an MDP action.

    Returns:
      next_state_idx, the index of the next state.
    """
    next_state_distribution = self.transition_model(state_idx, action_idx)
    next_state_idx = np.random.choice(next_state_distribution[:, 1],
                                      p=next_state_distribution[:, 0])
    return int(next_state_idx)

  @abc.abstractmethod
  def reward(self, state_idx: int, action_idx: int, *args, **kwargs) -> float:
    """Defines MDP reward function.

      Args:
        state_idx: Index of an MDP state.
        action_idx: Index of an MDP action.

      Returns:
        A scalar reward.
    """
    raise NotImplementedError

  @property
  def np_reward_model(self):
    """Returns reward model as a np ndarray."""
    # If already computed, return the computed value.
    # This model does not change after the MDP is defined.
    if self._np_reward_model is not None:
      return self._np_reward_model

    # Else: Compute using the reward method.
    # Set -inf to unreachable states to prevent an action from falling in them
    # Perhaps, we need to switch -inf to a large negative number later
    # numpy.nan_to_num() can be an option
    self._np_reward_model = np.full((self.num_states, self.num_actions),
                                    -np.inf)
    for state in tqdm(range(self.num_states)):
      if self.is_terminal(state):
        # there is no valid action at the terminal state
        # but set them to have 0 reward
        # in order to make planning algorithms work
        self._np_reward_model[state, :] = 0
      else:
        # for action in range(self.num_actions):
        for action in self.legal_actions(state):
          self._np_reward_model[state, action] = self.reward(state, action)
    return self._np_reward_model

  @abc.abstractmethod
  def is_terminal(self, state_idx: int):
    """
    Optional. Implement if needed.
    You can just define any transitions from terminal states
    to get back to itself with probability 1 and
    define the reward for one transition as 0 and others as -inf
    In this way, you can use numpy array for MDP.
    """
    return False

  @abc.abstractmethod
  def legal_actions(self, state_idx: int):
    """
    Optional. Implement if needed.
    You can just define any transitions of invalid actions
    to get back to itself with probability 1 and
    assign rewards for invalid transitions with -inf
    In this way, you can use numpy array for MDP.
    """
    return list(range(self.num_actions))


def v_value_from_q_value(q_value: np.ndarray) -> np.ndarray:
  """Computes V values given Q values.

  Args:
    q_value: A numpy 2-d array of Q values. First dimension should correspond
      to state, the second to action.

  Returns:
    value of a state, V(s), as a numpy 1-d array.
  """
  return q_value.max(axis=-1)


def q_value_from_v_value(
    v_value: np.ndarray,
    transition_model: Union[np.ndarray, sparse.COO],
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
) -> np.ndarray:
  """Computes V values given a policy.

  Args:
    v_value: value of a state, V(s), as a numpy 1-d array.
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.

  Returns:
    value of a state and action pair, Q(s,a), as a numpy 2-d array.
  """
  if isinstance(transition_model, sparse.COO):
    q_value = reward_model + discount_factor * sparse.tensordot(
        transition_model, np.nan_to_num(v_value), axes=(2, 0))
  else:
    q_value = reward_model + discount_factor * np.tensordot(
        transition_model, np.nan_to_num(v_value), axes=(2, 0))

  return q_value


def v_value_from_policy(
    policy: np.ndarray,
    transition_model: Union[np.ndarray, sparse.COO],
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    v_value_initial: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Computes V values given a policy.

  Args:
    policy: A policy. Coule be either deterministic or stochastic.
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    value of a state, V(s), as a numpy 1-d array.
  """
  num_states, num_actions, _ = transition_model.shape

  if policy.ndim == 1:
    stochastic_policy = np.zeros((num_states, num_actions))
    stochastic_policy[np.arange(num_states), policy] = 1.
  elif policy.ndim == 2:
    stochastic_policy = policy
  else:
    raise ValueError("Provided policy has incorrect dimension.")

  if v_value_initial is not None:
    assert v_value_initial.shape == (num_states, ), (
        "Initial V value has incorrect shape.")
    v_value = v_value_initial
  else:
    v_value = np.zeros((num_states))

  iteration_idx = 0
  delta_v = epsilon + 1.
  progress_bar = tqdm(total=max_iteration)
  while (iteration_idx < max_iteration) and (delta_v > epsilon):
    q_value = q_value_from_v_value(v_value, transition_model, reward_model,
                                   discount_factor)

    # replacing -inf (i.e., illegal state-action) with 0
    q_value[np.isneginf(q_value)] = 0

    new_v_value = np.sum(stochastic_policy * np.nan_to_num(q_value), axis=-1)

    delta_v = np.linalg.norm(new_v_value[:] - v_value[:])
    iteration_idx += 1
    v_value = new_v_value
    progress_bar.set_postfix({'delta': delta_v})
    progress_bar.update()
  progress_bar.close()

  return v_value


def q_value_from_policy(
    policy: np.ndarray,
    transition_model: Union[np.ndarray, sparse.COO],
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    v_value_initial: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Computes V values given a policy.

  Args:
    policy: A policy. Coule be either deterministic or stochastic.
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    value of a state and action pair, Q(s,a), as a numpy 2-d array.
  """

  v_value = v_value_from_policy(
      policy=policy,
      transition_model=transition_model,
      reward_model=reward_model,
      discount_factor=discount_factor,
      max_iteration=max_iteration,
      epsilon=epsilon,
      v_value_initial=v_value_initial,
  )

  q_value = q_value_from_v_value(
      v_value=v_value,
      transition_model=transition_model,
      reward_model=reward_model,
      discount_factor=discount_factor,
  )

  return q_value


def deterministic_policy_from_q_value(q_value: np.ndarray) -> np.ndarray:
  """Computes a deterministic policy given Q values.

  Args:
    q_value: A numpy 2-d array of Q values. First dimension should correspond
      to state, the second to action.

  Returns:
    action in a state, policy(s), as a numpy 1-d array.
  """
  return q_value.argmax(axis=-1)


def softmax_policy_from_q_value(q_value: np.ndarray,
                                temperature: float = 1.) -> np.ndarray:
  """Computes a stochastic softmax policy given Q values.

  Args:
    q_value: A numpy 2-d array of Q values. First dimension should correspond
      to state, the second to action.
    temperature: The temperature parameters while computing the softmax. For a
      high temperature, the policy will be uniform over action. For more
      details, see https://en.wikipedia.org/wiki/Softmax_function#Applications

  Returns:
    probability of an action in a state, policy(s,a), as a numpy 2-d array.
  """
  if temperature == 0:
    num_states, num_actions = q_value.shape
    policy = q_value.argmax(axis=-1)
    stochastic_policy = np.zeros((num_states, num_actions))
    stochastic_policy[np.arange(num_states), policy] = 1.
    return stochastic_policy
  else:
    return sc.softmax(np.nan_to_num(q_value) / temperature, axis=-1)
