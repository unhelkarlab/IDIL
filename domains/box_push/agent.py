from models.policy import CachedPolicyInterface
from domains.agent import AIAgent_Abstract
from domains.box_push.agent_model import (BoxPushAM, BoxPushAM_Alone,
                                          BoxPushAM_Together,
                                          BoxPushAM_EmptyMind,
                                          BoxPushAM_WebExp_Both)


class BoxPushAIAgent_Host(AIAgent_Abstract):

  def __init__(self, policy_model: CachedPolicyInterface) -> None:
    super().__init__(policy_model, has_mind=False)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_EmptyMind(policy_model)

  def init_latent(self, tup_states):
    'do nothing - a user should set the latent state manually'
    pass

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing - a user should set the latent state manually'
    pass


class BoxPushAIAgent_Team1(AIAgent_Abstract):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=0)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Together(agent_idx=self.agent_idx,
                              policy_model=policy_model)


class BoxPushAIAgent_Team2(AIAgent_Abstract):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=1)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Together(agent_idx=self.agent_idx,
                              policy_model=policy_model)


class BoxPushAIAgent_WebExp_Both_A2(AIAgent_Abstract):

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_WebExp_Both(policy_model=policy_model)


class BoxPushAIAgent_Indv1(AIAgent_Abstract):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=0)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Alone(self.agent_idx, policy_model)


class BoxPushAIAgent_Indv2(AIAgent_Abstract):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(policy_model, has_mind, agent_idx=1)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> BoxPushAM:
    return BoxPushAM_Alone(self.agent_idx, policy_model)
