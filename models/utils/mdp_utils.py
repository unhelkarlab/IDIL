"""Utilities for MDPs."""

from typing import Dict, Optional, Sequence
import numpy as np


class StateSpace:
  """Defines a state space."""
  def __init__(
      self,
      statespace: Optional[Sequence] = None,
      idx_to_state: Optional[Dict] = None,
      state_to_idx: Optional[Dict] = None,
  ):
    """Initializes a state space.

    Args:
      statespace: Optional; a sequence of states. Defaults to a state space with
        one state. The input sequence should not contain any duplicated element
      idx_to_state: Optional; a mapping from state indices to states.
      state_to_idx: Optional; a mapping from states to state indices.
    """

    # Defines the state space; default state space withe one element.
    self.statespace = statespace if statespace is not None else set([0])
    self.num_states = len(self.statespace)

    # check duplicates
    set_tmp = set()
    for elem in self.statespace:
      if elem in set_tmp:
        raise ValueError("Found duplicates in the statespace")
      set_tmp.add(elem)

    # creates a mapping from state index to state, and vice-versa
    if idx_to_state is None:
      self.idx_to_state = {
          state_idx: state
          for state_idx, state in enumerate(self.statespace)
      }
    else:
      self.idx_to_state = idx_to_state

    if state_to_idx is None:
      self.state_to_idx = {
          state: state_idx
          for state_idx, state in self.idx_to_state.items()
      }
    else:
      self.state_to_idx = state_to_idx


class NpStateSpace:
  """Defines a state space using numpy."""
  def __init__(
      self,
      np_idx_to_state: np.ndarray,
      np_state_to_idx: np.ndarray,
  ):
    """Initializes state space using numpy.

    Args:
      np_idx_to_state: a mapping from state indices to states.
      np_state_to_idx: a mapping from states to state indices.
    """

    self.np_idx_to_state = np_idx_to_state
    self.np_state_to_idx = np_state_to_idx
    self.num_states = self.np_idx_to_state.shape[0]


class ActionSpace:
  """Defines an action space."""
  def __init__(
      self,
      actionspace: Optional[Sequence] = None,
      idx_to_action: Optional[Dict] = None,
      action_to_idx: Optional[Dict] = None,
  ):
    """Initializes an action space.

    Args:
      actionspace: Optional; a sequence of actions. Defaults to a action space
          with one action. The input should not contain any duplicated element
      idx_to_action: Optional; a mapping from action indices to actions.
      action_to_idx: Optional; a mapping from actions to action indices.
    """

    # Defines the action space; default action space withe one element.
    self.actionspace = actionspace if actionspace is not None else set([0])
    self.num_actions = len(self.actionspace)

    # check duplicates
    set_tmp = set()
    for elem in self.actionspace:
      if elem in set_tmp:
        raise ValueError("Found duplicates in the actionspace")
      set_tmp.add(elem)

    # creates a mapping from action index to action, and vice-versa
    if idx_to_action is None:
      self.idx_to_action = {
          action_idx: action
          for action_idx, action in enumerate(self.actionspace)
      }
    else:
      self.idx_to_action = idx_to_action

    if action_to_idx is None:
      self.action_to_idx = {
          action: action_idx
          for action_idx, action in self.idx_to_action.items()
      }
    else:
      self.action_to_idx = action_to_idx


class NpActionSpace:
  """Defines an action space using numpy."""
  def __init__(
      self,
      np_idx_to_action: np.ndarray,
      np_action_to_idx: np.ndarray,
  ):
    """Initializes action space using numpy.

    Args:
      np_idx_to_action: a mapping from action indices to actions.
      np_action_to_idx: a mapping from actions to action indices.
    """

    self.np_idx_to_action = np_idx_to_action
    self.np_action_to_idx = np_action_to_idx
    self.num_actions = self.np_idx_to_action.shape[0]
