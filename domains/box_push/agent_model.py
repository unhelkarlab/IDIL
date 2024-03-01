from typing import Sequence, Optional
import numpy as np
from models.agent_model import AgentModel
from models.policy import PolicyInterface
from domains.box_push import conv_box_idx_2_state, BoxState
from domains.box_push.mdp import BoxPushMDP


def get_holding_box_and_floor_boxes(box_states, num_drops, num_goals):
  a1_box = -1
  a2_box = -1
  floor_boxes = []
  for idx in range(len(box_states)):
    state = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
    if state[0] == BoxState.WithAgent1:
      a1_box = idx
    elif state[0] == BoxState.WithAgent2:
      a2_box = idx
    elif state[0] == BoxState.WithBoth:
      a1_box = idx
      a2_box = idx
    elif state[0] in [BoxState.Original, BoxState.OnDropLoc]:
      floor_boxes.append(idx)

  return a1_box, a2_box, floor_boxes


def assumed_initial_mental_distribution(agent_idx: int, obstate_idx: int,
                                        mdp: BoxPushMDP):
  '''
      obstate_idx: absolute (task-perspective) state representation.
  '''
  box_states, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)

  num_drops = len(mdp.drops)
  num_goals = len(mdp.goals)
  a1_box, a2_box, valid_box = get_holding_box_and_floor_boxes(
      box_states, num_drops, num_goals)

  def get_np_bx_from_hold_state(my_box, mate_box):
    # it makes more sense to assume an agent is more likely to go to the goal
    P_ORIG = 0.1
    P_DROP = 0
    P_GOAL = 1 - P_ORIG - P_DROP

    np_bx = np.zeros(mdp.num_latents)
    if my_box >= 0:
      xidx = mdp.latent_space.state_to_idx[("origin", 0)]
      np_bx[xidx] = P_ORIG
      for idx in range(num_drops):
        xidx = mdp.latent_space.state_to_idx[("drop", idx)]
        np_bx[xidx] = P_DROP / num_drops
      for idx in range(num_goals):
        xidx = mdp.latent_space.state_to_idx[("goal", idx)]
        np_bx[xidx] = P_GOAL / num_goals
    else:
      num_valid_box = len(valid_box)
      if num_valid_box > 0:
        for idx in valid_box:
          xidx = mdp.latent_space.state_to_idx[("pickup", idx)]
          np_bx[xidx] = 1 / num_valid_box
      else:
        if mate_box >= 0:  # TODO: check > or >=
          xidx = mdp.latent_space.state_to_idx[("pickup", mate_box)]
          np_bx[xidx] = 1
        else:  # game finished, not meaningful state
          xidx = mdp.latent_space.state_to_idx[("goal", 0)]
          np_bx[xidx] = 1

    return np_bx

  if agent_idx == 0:
    return get_np_bx_from_hold_state(a1_box, a2_box)
  else:
    return get_np_bx_from_hold_state(a2_box, a1_box)


class BoxPushAM(AgentModel):

  def __init__(self,
               agent_idx: int,
               policy_model: Optional[PolicyInterface] = None) -> None:
    super().__init__(policy_model)
    self.agent_idx = agent_idx

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    '''
        assume agent1 and 2 has the same policy
        state_idx: absolute (task-perspective) state representation.
                    For here, we assume agent1 state and task state is the same
        '''
    mdp = self.get_reference_mdp()  # type: BoxPushMDP
    return assumed_initial_mental_distribution(self.agent_idx, obstate_idx, mdp)


class BoxPushAM_EmptyMind(BoxPushAM):

  def __init__(self, policy_model: Optional[PolicyInterface] = None) -> None:
    super().__init__(0, policy_model)

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    'should not be called'
    raise NotImplementedError

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    'should not be called'
    raise NotImplementedError


class BoxPushAM_Together(BoxPushAM):

  def get_np_Tx_team_impl(self,
                          latstate_idx,
                          my_box_cur,
                          mate_box_cur,
                          my_box_nxt,
                          mate_box_nxt,
                          mate_pos,
                          box_states_nxt,
                          valid_boxes_nxt,
                          p_change=0.1):
    mdp = self.get_reference_mdp()  # type: BoxPushMDP

    num_drops = len(mdp.drops)
    num_goals = len(mdp.goals)

    my_pickup = my_box_cur < 0 and my_box_nxt >= 0
    my_drop = my_box_cur >= 0 and my_box_nxt < 0

    num_valid_box = len(valid_boxes_nxt)
    np_Tx = np.zeros(self.policy_model.get_num_latent_states())
    if my_pickup:
      xidx = self.policy_model.conv_latent_to_idx(("goal", 0))
      np_Tx[xidx] = 1
      return np_Tx
    elif my_drop:
      if num_valid_box > 0:
        for idx in valid_boxes_nxt:
          xidx = self.policy_model.conv_latent_to_idx(("pickup", idx))
          np_Tx[xidx] = 1 / num_valid_box
        return np_Tx
    elif my_box_nxt < 0:
      latent = self.policy_model.conv_idx_to_latent(latstate_idx)
      # change latent based on teammate position
      if latent[0] == "pickup":
        min_idx = None
        dist_min = mdp.x_grid * mdp.y_grid
        for idx, bidx in enumerate(box_states_nxt):
          bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
          if bstate[0] == BoxState.Original:
            box_pos = mdp.boxes[idx]
            dist_tmp = abs(mate_pos[0] - box_pos[0]) + abs(mate_pos[1] -
                                                           box_pos[1])
            if dist_min > dist_tmp:
              dist_min = dist_tmp
              min_idx = idx
        if min_idx is not None and dist_min == 0 and latent[1] != min_idx:
          xidx = self.policy_model.conv_latent_to_idx(("pickup", min_idx))
          np_Tx[xidx] = p_change
          np_Tx[latstate_idx] = 1 - p_change
          return np_Tx

    np_Tx[latstate_idx] = 1
    return np_Tx

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    '''
    obstate_idx: absolute (task-perspective) state representation.
          Here, we assume agent1 state space and task state space is the same
    '''
    mdp = self.get_reference_mdp()  # type: BoxPushMDP

    num_drops = len(mdp.drops)
    num_goals = len(mdp.goals)

    box_states_cur, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)
    box_states_nxt, a1_pos, a2_pos = mdp.conv_mdp_sidx_to_sim_states(
        obstate_next_idx)

    a1_box_cur, a2_box_cur, _ = get_holding_box_and_floor_boxes(
        box_states_cur, num_drops, num_goals)
    a1_box_nxt, a2_box_nxt, valid_boxes = get_holding_box_and_floor_boxes(
        box_states_nxt, num_drops, num_goals)

    if self.agent_idx == 0:
      return self.get_np_Tx_team_impl(latstate_idx, a1_box_cur, a2_box_cur,
                                      a1_box_nxt, a2_box_nxt, a2_pos,
                                      box_states_nxt, valid_boxes, 0.1)
    else:
      return self.get_np_Tx_team_impl(latstate_idx, a2_box_cur, a1_box_cur,
                                      a2_box_nxt, a1_box_nxt, a1_pos,
                                      box_states_nxt, valid_boxes, 0.1)


class BoxPushAM_WebExp_Both(BoxPushAM_Together):

  def __init__(self, policy_model: Optional[PolicyInterface] = None) -> None:
    super().__init__(1, policy_model=policy_model)

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    '''
    obstate_idx: absolute (task-perspective) state representation.
          Here, we assume agent1 state space and task state space is the same
    '''
    mdp = self.get_reference_mdp()  # type: BoxPushMDP

    num_drops = len(mdp.drops)
    num_goals = len(mdp.goals)

    box_states_cur, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)
    box_states_nxt, a1_pos, a2_pos = mdp.conv_mdp_sidx_to_sim_states(
        obstate_next_idx)

    a1_box_cur, a2_box_cur, _ = get_holding_box_and_floor_boxes(
        box_states_cur, num_drops, num_goals)
    a1_box_nxt, a2_box_nxt, valid_boxes = get_holding_box_and_floor_boxes(
        box_states_nxt, num_drops, num_goals)

    if self.agent_idx == 0:
      return self.get_np_Tx_team_impl(latstate_idx, a1_box_cur, a2_box_cur,
                                      a1_box_nxt, a2_box_nxt, a2_pos,
                                      box_states_nxt, valid_boxes, 0.0)
    else:
      return self.get_np_Tx_team_impl(latstate_idx, a2_box_cur, a1_box_cur,
                                      a2_box_nxt, a1_box_nxt, a1_pos,
                                      box_states_nxt, valid_boxes, 0.0)


class BoxPushAM_Alone(BoxPushAM):

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    '''
    obstate_idx: absolute (task-perspective) state representation.
          Here, we assume agent1 state space and task state space is the same
    '''
    mdp = self.get_reference_mdp()  # type: BoxPushMDP

    num_drops = len(mdp.drops)
    num_goals = len(mdp.goals)

    box_states_cur, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)
    box_states_nxt, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_next_idx)

    a1_box_cur, a2_box_cur, _ = get_holding_box_and_floor_boxes(
        box_states_cur, num_drops, num_goals)
    a1_box_nxt, a2_box_nxt, valid_boxes = get_holding_box_and_floor_boxes(
        box_states_nxt, num_drops, num_goals)

    def get_np_Tx_indv_impl(my_box_cur, mate_box_cur, my_box_nxt, mate_box_nxt):
      my_pickup = my_box_cur < 0 and my_box_nxt >= 0
      my_drop = my_box_cur >= 0 and my_box_nxt < 0
      mate_pickup = mate_box_cur < 0 and mate_box_nxt >= 0

      num_valid_box = len(valid_boxes)
      np_Tx = np.zeros(self.policy_model.get_num_latent_states())
      if my_pickup:
        xidx = self.policy_model.conv_latent_to_idx(("goal", 0))
        np_Tx[xidx] = 1
        return np_Tx
      elif my_drop:
        if num_valid_box > 0:
          for idx in valid_boxes:
            xidx = self.policy_model.conv_latent_to_idx(("pickup", idx))
            np_Tx[xidx] = 1 / num_valid_box
          return np_Tx
        else:
          if mate_box_nxt >= 0:  # TODO: check > or >=
            xidx = self.policy_model.conv_latent_to_idx(
                ("pickup", mate_box_nxt))
            np_Tx[xidx] = 1
            return np_Tx
      elif mate_pickup:
        latent = self.policy_model.conv_idx_to_latent(latstate_idx)
        if my_box_nxt < 0 and latent[0] == "pickup" and latent[
            1] == mate_box_nxt:
          if num_valid_box > 0:
            for idx in valid_boxes:
              xidx = self.policy_model.conv_latent_to_idx(("pickup", idx))
              np_Tx[xidx] = 1 / num_valid_box
            return np_Tx

      np_Tx[latstate_idx] = 1
      return np_Tx

    if self.agent_idx == 0:
      return get_np_Tx_indv_impl(a1_box_cur, a2_box_cur, a1_box_nxt, a2_box_nxt)
    else:
      return get_np_Tx_indv_impl(a2_box_cur, a1_box_cur, a2_box_nxt, a1_box_nxt)
