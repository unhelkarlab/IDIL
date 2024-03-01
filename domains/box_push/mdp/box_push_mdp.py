from typing import Sequence, Tuple, Union
import abc
import numpy as np
from models.utils.mdp_utils import StateSpace
from models.mdp import LatentMDP
from domains.box_push import (BoxState, conv_box_state_2_idx,
                              conv_box_idx_2_state, get_possible_latent_states)


class BoxPushMDP(LatentMDP):

  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    super().__init__(use_sparse=True)

  @abc.abstractmethod
  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    raise NotImplementedError

  def map_to_str(self):
    BASE36 = 36
    assert self.x_grid < BASE36 and self.y_grid < BASE36

    x_36 = np.base_repr(self.x_grid, BASE36)
    y_36 = np.base_repr(self.y_grid, BASE36)

    np_map = np.zeros((self.x_grid, self.y_grid), dtype=int)
    if self.boxes:
      np_map[tuple(zip(*self.boxes))] = 1
    if self.goals:
      np_map[tuple(zip(*self.goals))] = 2
    if self.walls:
      np_map[tuple(zip(*self.walls))] = 3
    if self.drops:
      np_map[tuple(zip(*self.drops))] = 4

    map_5 = "".join(np_map.reshape((-1, )).astype(str))
    map_int = int(map_5, base=5)
    map_36 = np.base_repr(map_int, base=BASE36)

    tup_map = (x_36, y_36, map_36)

    return "%s_%s_%s" % tup_map

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    list_grid = []
    for i in range(self.x_grid):
      for j in range(self.y_grid):
        state = (i, j)
        if state not in self.walls:
          list_grid.append(state)
    self.pos1_space = StateSpace(statespace=list_grid)
    self.pos2_space = StateSpace(statespace=list_grid)
    self.dict_factored_statespace = {0: self.pos1_space, 1: self.pos2_space}

    box_states = self.get_possible_box_states()

    for dummy_i in range(len(self.boxes)):
      self.dict_factored_statespace[dummy_i +
                                    2] = StateSpace(statespace=box_states)

    self.dummy_states = None

  def init_latentspace(self):
    latent_states = get_possible_latent_states(len(self.boxes), len(self.drops),
                                               len(self.goals))
    self.latent_space = StateSpace(latent_states)

  def is_terminal(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      if box_state[0] != BoxState.OnGoalLoc:
        return False
    return True

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        holding_box = idx

    if holding_box >= 0 and a1_pos != a2_pos:  # illegal state
      return []

    return super().legal_actions(state_idx)

  @abc.abstractmethod
  def get_possible_box_states(
      self) -> Sequence[Tuple[BoxState, Union[int, None]]]:
    raise NotImplementedError

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    box_states, pos1, pos2 = tup_states
    len_s_space = len(self.dict_factored_statespace)
    pos1_idx = self.pos1_space.state_to_idx[pos1]
    pos2_idx = self.pos2_space.state_to_idx[pos2]
    list_states = [int(pos1_idx), int(pos2_idx)]
    for idx in range(2, len_s_space):
      box_sidx_n = box_states[idx - 2]
      box_state_n = conv_box_idx_2_state(box_sidx_n, len(self.drops),
                                         len(self.goals))
      box_sidx = self.dict_factored_statespace[idx].state_to_idx[box_state_n]
      list_states.append(int(box_sidx))

    return self.conv_state_to_idx(tuple(list_states))

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    pos1 = self.pos1_space.idx_to_state[state_vec[0]]
    pos2 = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(conv_box_state_2_idx(box_state, len(self.drops)))

    return box_states, pos1, pos2

  def conv_mdp_aidx_to_sim_actions(self, action_idx):
    vector_aidx = self.conv_idx_to_action(action_idx)
    list_actions = []
    for idx, aidx in enumerate(vector_aidx):
      list_actions.append(
          self.dict_factored_actionspace[idx].idx_to_action[aidx])
    return tuple(list_actions)

  def conv_sim_actions_to_mdp_aidx(self, tuple_actions):
    list_aidx = []
    for idx, act in enumerate(tuple_actions):
      list_aidx.append(self.dict_factored_actionspace[idx].action_to_idx[act])

    return self.np_action_to_idx[tuple(list_aidx)]
