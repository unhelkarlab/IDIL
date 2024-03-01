from typing import Tuple
from enum import Enum
from models.utils.mdp_utils import ActionSpace


class EventType(Enum):
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3
  STAY = 4
  HOLD = 5
  UNHOLD = HOLD
  SET_LATENT = 6


class BoxState(Enum):
  Original = 0
  WithAgent1 = 1
  WithAgent2 = 2
  WithBoth = 3
  OnDropLoc = 4
  OnGoalLoc = 5
  # overloaded for relative box states for individual MDPs
  WithMe = WithAgent1
  WithTeammate = WithAgent2


AGENT_ACTIONSPACE = ActionSpace([EventType(idx) for idx in range(6)])


def get_possible_latent_states(num_boxes, num_drops, num_goals):
  latent_states = []
  for idx in range(num_boxes):
    latent_states.append(("pickup", idx))

  latent_states.append(("origin", 0))  # drop at its original position
  for idx in range(num_drops):
    latent_states.append(("drop", idx))
  for idx in range(num_goals):
    latent_states.append(("goal", idx))

  return latent_states


def conv_box_idx_2_state(state_idx, num_drops, num_goals=None):
  if state_idx >= 0 and state_idx < 4:
    return (BoxState(state_idx), None)
  elif state_idx >= 4 and state_idx < 4 + num_drops:
    return (BoxState(4), state_idx - 4)
  elif state_idx >= 4 + num_drops:
    return (BoxState(5), state_idx - 4 - num_drops)

  raise ValueError


def conv_box_state_2_idx(state: Tuple[BoxState, int], num_drops):
  if state[0].value < 4:
    return state[0].value
  elif state[0] == BoxState.OnDropLoc:
    return 4 + state[1]
  elif state[0] == BoxState.OnGoalLoc:
    return 4 + num_drops + state[1]

  raise ValueError
