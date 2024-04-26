from typing import Optional, Sequence
import numpy as np
from hcair_models.policy import CachedPolicyInterface, PolicyInterface
from hcair_models.agent_model import AgentModel
from hcair_domains.agent import AIAgent_Abstract
from hcair_domains.box_push import conv_box_idx_2_state, BoxState
from hcair_domains.cleanup_single.mdp import MDPCleanupSingle


class AM_CleanupSingle(AgentModel):

  def __init__(self, policy_model: Optional[PolicyInterface] = None) -> None:
    super().__init__(policy_model)

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    '''
        state_idx: absolute (task-perspective) state representation.
    '''
    mdp = self.get_reference_mdp()  # type: MDPCleanupSingle

    box_states, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)

    np_bx = np.zeros(mdp.num_latents)
    num_boxes = len(box_states)
    for idx in range(num_boxes):
      xidx = mdp.latent_space.state_to_idx[("pickup", idx)]
      np_bx[xidx] = 1 / num_boxes

    return np_bx

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    mdp = self.get_reference_mdp()  # type: MDPCleanupSingle

    box_state_cur, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)
    box_state_nxt, pos = mdp.conv_mdp_sidx_to_sim_states(obstate_next_idx)

    num_drops = len(mdp.drops)

    def get_holding_box(box_states):
      holding_box = -1
      floor_boxes = []
      for idx in range(len(box_states)):
        state = conv_box_idx_2_state(box_states[idx], num_drops)
        if state[0] == BoxState.WithAgent1:
          holding_box = idx
        elif state[0] in [BoxState.Original, BoxState.OnDropLoc]:
          floor_boxes.append(idx)
      return holding_box, floor_boxes

    holding_box_cur, _ = get_holding_box(box_state_cur)
    holding_box_nxt, floor_boxes = get_holding_box(box_state_nxt)

    has_picked_up = holding_box_cur < 0 and holding_box_nxt >= 0
    has_dropped = holding_box_cur >= 0 and holding_box_nxt < 0
    num_floor_boxes = len(floor_boxes)
    np_Tx = np.zeros(self.policy_model.get_num_latent_states())
    if has_picked_up:
      xidx = self.policy_model.conv_latent_to_idx(("goal", 0))
      np_Tx[xidx] = 1
      return np_Tx
    elif has_dropped:
      if num_floor_boxes > 0:
        for idx in floor_boxes:
          xidx = self.policy_model.conv_latent_to_idx(("pickup", idx))
          np_Tx[xidx] = 1 / num_floor_boxes
        return np_Tx

    np_Tx[latstate_idx] = 1
    return np_Tx


class Agent_CleanupSingle(AIAgent_Abstract):

  def __init__(self, policy_model: CachedPolicyInterface) -> None:
    super().__init__(policy_model, True, agent_idx=0)

  def _create_agent_model(self, policy_model: CachedPolicyInterface):
    return AM_CleanupSingle(policy_model)
