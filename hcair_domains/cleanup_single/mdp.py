import numpy as np
from hcair_models.utils.mdp_utils import StateSpace
from hcair_models.mdp import LatentMDP
from hcair_domains.box_push import (BoxState, conv_box_state_2_idx,
                                    AGENT_ACTIONSPACE, EventType,
                                    conv_box_idx_2_state)
from hcair_domains.cleanup_single.transition import transition_single_agent
from .define import get_possible_latent_states


class MDPCleanupSingle(LatentMDP):

  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, init_pos,
               **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    self.init_pos = init_pos
    super().__init__(use_sparse=True)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    box_states, agent_pos = self.conv_mdp_sidx_to_sim_states(state_idx)
    agent_action, = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = transition_single_agent(box_states, agent_pos,
                                              agent_action, self.boxes,
                                              self.goals, self.walls,
                                              self.drops, self.x_grid,
                                              self.y_grid, self.init_pos)

    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, agent_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx([box_states_list, agent_pos_n])
      # assume a2 choose an action uniformly
      map_next_state[sidx_n] = (map_next_state.get(sidx_n, 0) + p)

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

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
    self.pos_space = StateSpace(statespace=list_grid)
    self.dict_factored_statespace = {0: self.pos_space}

    box_states = self.get_possible_box_states()

    for idx in range(len(self.boxes)):
      self.dict_factored_statespace[idx + 1] = StateSpace(statespace=box_states)

    self.dummy_states = None

  def init_latentspace(self):
    latent_states = get_possible_latent_states(len(self.boxes), len(self.drops),
                                               len(self.goals))
    self.latent_space = StateSpace(latent_states)

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    self.my_act_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.my_act_space}

  def is_terminal(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    for idx in range(1, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      if box_state[0] != BoxState.OnGoalLoc:
        return False
    return True

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    return super().legal_actions(state_idx)

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithAgent1, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states

  def conv_sim_states_to_mdp_sidx(self, tup_states):
    box_states, agent_pos = tup_states
    len_s_space = len(self.dict_factored_statespace)
    pos_idx = self.pos_space.state_to_idx[agent_pos]
    list_states = [int(pos_idx)]
    for idx in range(1, len_s_space):
      box_sidx_n = box_states[idx - 1]
      box_state_n = conv_box_idx_2_state(box_sidx_n, len(self.drops),
                                         len(self.goals))
      box_sidx = self.dict_factored_statespace[idx].state_to_idx[box_state_n]
      list_states.append(int(box_sidx))

    return self.conv_state_to_idx(tuple(list_states))

  def conv_mdp_sidx_to_sim_states(self, state_idx):
    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    agent_pos = self.pos_space.idx_to_state[state_vec[0]]

    box_states = []
    for idx in range(1, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(conv_box_state_2_idx(box_state, len(self.drops)))

    return box_states, agent_pos

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

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    agent_pos = self.pos_space.idx_to_state[state_vec[0]]

    box_states = []
    for idx in range(1, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    agent_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithAgent1:
        holding_box = idx

    panelty = -1

    if latent[0] == "pickup":
      bidx = latent[1]
      desired_loc = self.boxes[bidx]
      if holding_box >= 0:
        if holding_box != bidx and agent_act == EventType.HOLD:
          return -np.inf
        reward = -((agent_pos[0] - desired_loc[0])**2 +
                   (agent_pos[1] - desired_loc[1])**2)
        return reward
      else:
        if agent_pos == desired_loc and agent_act == EventType.HOLD:
          return 0
        if agent_pos != desired_loc and agent_act == EventType.HOLD:
          return -np.inf
    else:  # latent[0] == "goal"
      desired_loc = self.goals[latent[1]]
      if holding_box >= 0:  # drop the box
        if agent_pos == desired_loc and agent_act == EventType.UNHOLD:
          return 0
        if agent_pos != desired_loc and agent_act == EventType.UNHOLD:
          return -np.inf
      else:  # not having a box
        if agent_act == EventType.HOLD:
          return -np.inf
        reward = -((agent_pos[0] - desired_loc[0])**2 +
                   (agent_pos[1] - desired_loc[1])**2) + 1
        return reward

    return panelty
