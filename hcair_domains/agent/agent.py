import abc
from hcair_models.policy import CachedPolicyInterface
from hcair_models.agent_model import AgentModel
from hcair_models.mdp import LatentMDP


class SimulatorAgent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, has_mind: bool, has_policy: bool) -> None:
    self.bool_mind = has_mind
    self.bool_policy = has_policy

  def has_mind(self):
    return self.bool_mind

  def has_policy(self):
    return self.bool_policy

  @abc.abstractmethod
  def init_latent(self, tup_states):
    raise NotImplementedError

  @abc.abstractmethod
  def get_current_latent(self):
    raise NotImplementedError

  @abc.abstractmethod
  def get_action(self, tup_states):
    raise NotImplementedError

  @abc.abstractmethod
  def set_latent(self, latent):
    'to set latent manually'
    raise NotImplementedError

  @abc.abstractmethod
  def set_action(self, action):
    'to set what to do as next actions manually'
    raise NotImplementedError

  @abc.abstractmethod
  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    raise NotImplementedError


class InteractiveAgent(SimulatorAgent):

  def __init__(self, start_latent=None) -> None:
    super().__init__(has_mind=False, has_policy=False)
    self.current_latent = None
    self.start_latent = start_latent
    self.action_queue = []

  def init_latent(self, tup_states):
    self.current_latent = self.start_latent

  def get_current_latent(self):
    return self.current_latent

  def get_action(self, tup_states):
    if len(self.action_queue) == 0:
      return None

    return self.action_queue.pop()

  def set_latent(self, latent):
    self.current_latent = latent

  def set_action(self, action):
    self.action_queue = [action]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'do nothing'
    pass


class AIAgent_Abstract(SimulatorAgent):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(has_mind=has_mind, has_policy=True)
    self.agent_idx = agent_idx
    self.agent_model = self._create_agent_model(policy_model)
    self.manual_action = None

  @abc.abstractmethod
  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AgentModel:
    'Should be implemented at inherited method'
    raise NotImplementedError

  def init_latent(self, tup_states):
    mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
    sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)

    self.agent_model.set_init_mental_state_idx(sidx)

  def get_current_latent(self):
    if self.agent_model.is_current_latent_valid():
      return self.conv_idx_to_latent(self.agent_model.current_latent)
    else:
      return None

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
    sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)
    tup_aidx = self.agent_model.get_action_idx(sidx)
    return self.agent_model.policy_model.conv_idx_to_action(tup_aidx)[0]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'tup_actions: tuple of actions'

    mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
    sidx_cur = mdp.conv_sim_states_to_mdp_sidx(tup_cur_state)
    sidx_nxt = mdp.conv_sim_states_to_mdp_sidx(tup_nxt_state)

    list_aidx = []
    for idx, act in enumerate(tup_actions):
      if act is None:
        list_aidx.append(None)
      else:
        list_aidx.append(mdp.dict_factored_actionspace[idx].action_to_idx[act])

    self.agent_model.update_mental_state_idx(sidx_cur, tuple(list_aidx),
                                             sidx_nxt)

  def set_latent(self, latent):
    xidx = self.conv_latent_to_idx(latent)
    self.agent_model.set_init_mental_state_idx(None, xidx)

  def set_action(self, action):
    self.manual_action = action

  def get_action_distribution(self, state_idx, latent_idx):
    return self.agent_model.policy_model.policy(state_idx, latent_idx)

  def get_next_latent_distribution(self, latent_idx, state_idx,
                                   tuple_action_idx, next_state_idx):
    return self.agent_model.transition_mental_state(latent_idx, state_idx,
                                                    tuple_action_idx,
                                                    next_state_idx)

  def get_initial_latent_distribution(self, state_idx):
    return self.agent_model.initial_mental_distribution(state_idx)

  def conv_idx_to_latent(self, latent_idx):
    return self.agent_model.policy_model.conv_idx_to_latent(latent_idx)

  def conv_latent_to_idx(self, latent):
    return self.agent_model.policy_model.conv_latent_to_idx(latent)
