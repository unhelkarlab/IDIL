from .mdp_env.env_from_mdp import EnvFromMDP
from hcair_domains.box_push.maps import EXP1_MAP
from hcair_domains.box_push.policy import (BoxPushPolicyTeamExp1,
                                           BoxPushPolicyIndvExp1)
from hcair_domains.box_push.simulator import (BoxPushSimulator_AlwaysTogether,
                                              BoxPushSimulator_AlwaysAlone)
from hcair_domains.box_push.mdp import (BoxPushTeamMDP_AlwaysTogether,
                                        BoxPushTeamMDP_AlwaysAlone,
                                        BoxPushAgentMDP_AlwaysAlone)
from hcair_domains.box_push.agent import (BoxPushAIAgent_Team2,
                                          BoxPushAIAgent_Indv2)
from gym import spaces

TEMPERATURE = 0.3


class EnvBoxPush(EnvFromMDP):

  def __init__(self, game_map, mdp_task, robot_agent):

    init_bstate = [0] * len(game_map["boxes"])
    a1_init = game_map["a1_init"]
    a2_init = game_map["a2_init"]

    init_sidx = mdp_task.conv_sim_states_to_mdp_sidx(
        [init_bstate, a1_init, a2_init])

    self.robot_agent = robot_agent

    super().__init__(mdp_task, [init_sidx], use_central_action=True)

    # redefine action space
    self.action_space = spaces.Discrete(mdp_task.a1_a_space.num_actions)

  def step(self, human_aidx):
    mdp = self.mdp  # type: BoxPushTeamMDP_AlwaysTogether
    sim_state = self.mdp.conv_mdp_sidx_to_sim_states(self.cur_state)
    # cur_sidx = self.cur_state

    robot_action = self.robot_agent.get_action(sim_state)
    robot_aidx = mdp.a2_a_space.action_to_idx[robot_action]

    human_action = mdp.a1_a_space.idx_to_action[int(human_aidx)]

    action = mdp.conv_action_to_idx((human_aidx, robot_aidx))
    next_sidx, reward, done, info = super().step(action)

    next_sim_state = self.mdp.conv_mdp_sidx_to_sim_states(next_sidx)
    self.robot_agent.update_mental_state(sim_state,
                                         (human_action, robot_action),
                                         next_sim_state)

    return next_sidx, reward, done, info

  def reset(self):
    self.cur_state = super().reset()
    sim_state = self.mdp.conv_mdp_sidx_to_sim_states(self.cur_state)
    self.robot_agent.init_latent(sim_state)
    return self.cur_state


class EnvMovers_v0(EnvBoxPush):

  def __init__(self):
    game_map = EXP1_MAP
    mdp_task = BoxPushTeamMDP_AlwaysTogether(**game_map)
    robot_policy = BoxPushPolicyTeamExp1(mdp_task, TEMPERATURE,
                                         BoxPushSimulator_AlwaysTogether.AGENT2)
    robot_agent = BoxPushAIAgent_Team2(robot_policy)

    super().__init__(game_map, mdp_task, robot_agent)


class EnvCleanup_v0(EnvBoxPush):

  def __init__(self):
    game_map = EXP1_MAP
    mdp_task = BoxPushTeamMDP_AlwaysAlone(**game_map)
    mdp_agent = BoxPushAgentMDP_AlwaysAlone(**game_map)

    robot_policy = BoxPushPolicyIndvExp1(mdp_task, mdp_agent, TEMPERATURE,
                                         BoxPushSimulator_AlwaysAlone.AGENT2)
    robot_agent = BoxPushAIAgent_Indv2(robot_policy)

    super().__init__(game_map, mdp_task, robot_agent)
