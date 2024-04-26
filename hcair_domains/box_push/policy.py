from abc import abstractmethod
from typing import Sequence
import os
import numpy as np
from hcair_models.policy import CachedPolicyInterface, PolicyInterface
from hcair_models.mdp import LatentMDP
from hcair_domains.box_push.mdp import (BoxPushTeamMDP, BoxPushAgentMDP,
                                        BoxPushTeamMDP_AlwaysTogether,
                                        get_agent_switched_boxstates)

policy_exp1_list = []
policy_indv_list = []
policy_test_agent_list = []
policy_test_team_list = []


class BoxPushPolicyTeamExp1(CachedPolicyInterface):

  def __init__(self, mdp: BoxPushTeamMDP_AlwaysTogether, temperature: float,
               agent_idx: int) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_team_")
    str_fileprefix += mdp.map_to_str() + "_"
    super().__init__(mdp, str_fileprefix, policy_exp1_list, temperature,
                     (agent_idx, ))
    # TODO: check if mdp has the same configuration as EXP1_MAP


class PolicyFromIdenticalAgentMDP(PolicyInterface):
  '''
  This class can be used when all agents are identical so we just need a
  representative policy of them but need to convert states from the agent
  perspective to the task perspective.
  '''

  def __init__(self, task_mdp: LatentMDP, agent_idx: int) -> None:
    super().__init__(task_mdp)
    self.agent_idx = agent_idx

    # agent policy should be defined at child class
    self.agent_policy = None  # type: PolicyInterface

  @abstractmethod
  def _convert_task_state_2_agent_state(self, obstate_idx):
    raise NotImplementedError

  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    agent_obstate = self._convert_task_state_2_agent_state(obstate_idx)

    # all agents are assumed to have identical actions and latent states
    return self.agent_policy.policy(agent_obstate, latstate_idx)

  def get_action(self, obstate_idx: int, latstate_idx: int) -> Sequence[int]:
    agent_obstate = self._convert_task_state_2_agent_state(obstate_idx)

    # all agents are assumed to have identical actions and latent states
    return self.agent_policy.get_action(agent_obstate, latstate_idx)

  def conv_idx_to_action(self, tuple_aidx: Sequence[int]) -> Sequence:
    return self.agent_policy.conv_idx_to_action(tuple_aidx)

  def conv_action_to_idx(self, tuple_actions: Sequence) -> Sequence[int]:
    return self.agent_policy.conv_action_to_idx(tuple_actions)

  def get_num_actions(self):
    return self.agent_policy.get_num_actions()

  def get_num_latent_states(self):
    return self.agent_policy.get_num_latent_states()

  def conv_idx_to_latent(self, latent_idx: int):
    return self.agent_policy.conv_idx_to_latent(latent_idx)

  def conv_latent_to_idx(self, latent_state):
    return self.agent_policy.conv_latent_to_idx(latent_state)


class PolicyFromIdenticalAgentMDP_BoxPush(PolicyFromIdenticalAgentMDP):

  def _convert_task_state_2_agent_state(self, obstate_idx):
    box_states, a1_pos, a2_pos = self.mdp.conv_mdp_sidx_to_sim_states(
        obstate_idx)

    pos_1 = a1_pos
    pos_2 = a2_pos
    bstate = box_states
    if self.agent_idx == 1:
      pos_1 = a2_pos
      pos_2 = a1_pos
      bstate = get_agent_switched_boxstates(box_states, len(self.mdp.drops),
                                            len(self.mdp.goals))
    return self.agent_policy.mdp.conv_sim_states_to_mdp_sidx(
        [bstate, pos_1, pos_2])


class BoxPushPolicyIndvExp1(PolicyFromIdenticalAgentMDP_BoxPush):

  def __init__(self, task_mdp: BoxPushTeamMDP, agent_mdp: BoxPushAgentMDP,
               temperature: float, agent_idx: int) -> None:
    super().__init__(task_mdp, agent_idx)

    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_indv_")
    str_fileprefix += task_mdp.map_to_str() + "_"
    # In this cached policy, states are represented w.r.t agent1.
    # We need to convert states in task-mdp into states in agent1-mdp.
    self.agent_policy = CachedPolicyInterface(agent_mdp, str_fileprefix,
                                              policy_indv_list, temperature)


class BoxPushPolicyTeamTest(CachedPolicyInterface):

  def __init__(self, mdp: BoxPushTeamMDP, temperature: float,
               agent_idx: int) -> None:
    super().__init__(mdp, "", policy_test_team_list, temperature, (agent_idx, ))


class BoxPushPolicyIndvTest_New(PolicyFromIdenticalAgentMDP_BoxPush):

  def __init__(self, task_mdp: BoxPushTeamMDP, agent_mdp: BoxPushAgentMDP,
               temperature: float, agent_idx: int) -> None:
    super().__init__(task_mdp, agent_idx)

    self.agent_policy = CachedPolicyInterface(agent_mdp, "",
                                              policy_test_agent_list,
                                              temperature)
