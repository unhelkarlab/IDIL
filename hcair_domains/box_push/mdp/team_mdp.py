import numpy as np
from hcair_domains.box_push import BoxState, EventType, AGENT_ACTIONSPACE
from hcair_domains.box_push.transition import (transition_always_together,
                                               transition_always_alone)
from hcair_domains.box_push.mdp import BoxPushMDP


class BoxPushTeamMDP(BoxPushMDP):

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    self.a1_a_space = AGENT_ACTIONSPACE
    self.a2_a_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    box_states, a1_pos, a2_pos = self.conv_mdp_sidx_to_sim_states(state_idx)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = self._transition_impl(box_states, a1_pos, a2_pos, act1,
                                            act2)
    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, a1_pos_n, a2_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [box_states_list, a1_pos_n, a2_pos_n])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)


class BoxPushTeamMDP_AlwaysTogether(BoxPushTeamMDP):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_always_together(box_states, a1_pos, a2_pos, a1_action,
                                      a2_action, self.boxes, self.goals,
                                      self.walls, self.drops, self.x_grid,
                                      self.y_grid)

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithBoth, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        holding_box = idx

    panelty = -1

    if latent[0] == "pickup":
      # if they are already holding a box,
      # set every action but stay as illegal
      if holding_box >= 0:
        if act1 == EventType.STAY and act2 == EventType.STAY:
          return 0
        else:
          return -np.inf
      # if they are not holding a box,
      # give a reward when pickup the target box
      else:
        if (a1_pos == a2_pos and a1_pos == self.boxes[latent[1]]
            and act1 == EventType.HOLD and act2 == EventType.HOLD):
          return 100

        # if get close to the target, don't deduct
        box_pos = self.boxes[latent[1]]
        dist1 = abs(a1_pos[0] - box_pos[0]) + abs(a1_pos[1] - box_pos[1])
        dist2 = abs(a2_pos[0] - box_pos[0]) + abs(a2_pos[1] - box_pos[1])
        panelty += 1 / (dist1 + dist2 + 1)
    elif holding_box >= 0:  # not "pickup" and holding a box --> drop the box
      desired_loc = None
      if latent[0] == "origin":
        desired_loc = self.boxes[holding_box]
      elif latent[0] == "drop":
        desired_loc = self.drops[latent[1]]
      else:  # latent[0] == "goal"
        desired_loc = self.goals[latent[1]]

      if (a1_pos == a2_pos and a1_pos == desired_loc
          and act1 == EventType.UNHOLD and act2 == EventType.UNHOLD):
        return 100
    else:  # "drop the box" but not having a box (illegal state)
      if act1 == EventType.STAY and act2 == EventType.STAY:
        return 0
      else:
        return -np.inf

    return panelty


class BoxPushTeamMDP_AlwaysAlone(BoxPushTeamMDP):

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_always_alone(box_states, a1_pos, a2_pos, a1_action,
                                   a2_action, self.boxes, self.goals,
                                   self.walls, self.drops, self.x_grid,
                                   self.y_grid)

  def get_possible_box_states(self):
    box_states = [(BoxState(idx), None) for idx in range(3)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states
