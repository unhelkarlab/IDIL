from typing import Hashable, Tuple
from apps.app import AppInterface
from hcair_domains.box_push import EventType, BoxState, conv_box_idx_2_state
import hcair_domains.box_push.maps as bp_maps
import hcair_domains.box_push.simulator as bp_sim
import hcair_domains.box_push.mdp as bp_mdp
import hcair_domains.box_push.policy as bp_policy
import hcair_domains.box_push.agent as bp_agent

IS_MOVERS = True

GAME_MAP = bp_maps.EXP1_MAP
BoxPushPolicyTeam = bp_policy.BoxPushPolicyTeamExp1
BoxPushPolicyIndv = bp_policy.BoxPushPolicyIndvExp1

if IS_MOVERS:
  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysTogether
else:
  BoxPushSimulator = bp_sim.BoxPushSimulator_AlwaysAlone


class BoxPushApp(AppInterface):

  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    'define game related variables and objects'
    GAME_ENV_ID = 0
    self.x_grid = GAME_MAP["x_grid"]
    self.y_grid = GAME_MAP["y_grid"]
    self.game = BoxPushSimulator(GAME_ENV_ID)
    self.game.max_steps = 200
    temperature = 0.3
    if IS_MOVERS:
      task_mdp = bp_mdp.BoxPushTeamMDP_AlwaysTogether(**GAME_MAP)
      policy1 = BoxPushPolicyTeam(task_mdp,
                                  temperature=temperature,
                                  agent_idx=0)
      policy2 = BoxPushPolicyTeam(task_mdp,
                                  temperature=temperature,
                                  agent_idx=1)
      agent1 = bp_agent.BoxPushAIAgent_Team1(policy1)
      agent2 = bp_agent.BoxPushAIAgent_Team2(policy2)
    else:
      task_mdp = bp_mdp.BoxPushTeamMDP_AlwaysAlone(**GAME_MAP)
      agent_mdp = bp_mdp.BoxPushAgentMDP_AlwaysAlone(**GAME_MAP)
      policy1 = BoxPushPolicyIndv(task_mdp,
                                  agent_mdp,
                                  temperature=temperature,
                                  agent_idx=0)
      policy2 = BoxPushPolicyIndv(task_mdp,
                                  agent_mdp,
                                  temperature=temperature,
                                  agent_idx=1)
      agent1 = bp_agent.BoxPushAIAgent_Indv1(policy1)
      agent2 = bp_agent.BoxPushAIAgent_Indv2(policy2)

    self.game.init_game(**GAME_MAP)
    self.game.set_autonomous_agent(agent1=agent1, agent2=agent2)

  def _init_gui(self):
    self.main_window.title("Box Push")
    self.canvas_width = 300
    self.canvas_height = 300
    super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    agent_id = None
    action = None
    value = None
    # agent1 move
    if key_sym == "Left":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.LEFT
    elif key_sym == "Right":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.RIGHT
    elif key_sym == "Up":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.UP
    elif key_sym == "Down":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.DOWN
    elif key_sym == "p":
      agent_id = BoxPushSimulator.AGENT1
      action = EventType.HOLD
    # agent2 move
    elif key_sym == "a":
      agent_id = BoxPushSimulator.AGENT2
      action = EventType.LEFT
    elif key_sym == "d":
      agent_id = BoxPushSimulator.AGENT2
      action = EventType.RIGHT
    elif key_sym == "w":
      agent_id = BoxPushSimulator.AGENT2
      action = EventType.UP
    elif key_sym == "s":
      agent_id = BoxPushSimulator.AGENT2
      action = EventType.DOWN
    elif key_sym == "f":
      agent_id = BoxPushSimulator.AGENT2
      action = EventType.HOLD

    return (agent_id, action, value)

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    return (None, None, None)

  def _update_canvas_scene(self):
    data = self.game.get_env_info()
    box_states = data["box_states"]
    boxes = data["boxes"]
    drops = data["drops"]
    goals = data["goals"]
    walls = data["walls"]
    a1_pos = data["a1_pos"]
    a2_pos = data["a2_pos"]

    x_unit = int(self.canvas_width / self.x_grid)
    y_unit = int(self.canvas_height / self.y_grid)

    self.clear_canvas()
    for coord in boxes:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gray")

    for coord in goals:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gold")

    for coord in walls:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "black")

    for coord in drops:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gray")

    a1_hold = False
    a2_hold = False
    for bidx, sidx in enumerate(box_states):
      state = conv_box_idx_2_state(sidx, len(drops), len(goals))
      box = None
      box_color = "green2"
      if state[0] == BoxState.Original:
        box = boxes[bidx]
      elif state[0] == BoxState.WithAgent1:
        box = a1_pos
        a1_hold = True
        box_color = "green4"
      elif state[0] == BoxState.WithAgent2:
        box = a2_pos
        a2_hold = True
        box_color = "green4"
      elif state[0] == BoxState.WithBoth:
        box = a1_pos
        a1_hold = True
        a2_hold = True
        box_color = "green4"
      elif state[0] == BoxState.OnDropLoc:
        box = drops[state[1]]

      if box is not None:
        self.create_rectangle(box[0] * x_unit, box[1] * y_unit,
                              (box[0] + 1) * x_unit, (box[1] + 1) * y_unit,
                              box_color)

    a1_color = "blue"
    if a1_hold:
      a1_color = "dark slate blue"
    self.create_circle((a1_pos[0] + 0.5) * x_unit, (a1_pos[1] + 0.5) * y_unit,
                       x_unit * 0.5, a1_color)

    a2_color = "red"
    if a2_hold:
      a2_color = "indian red"
    self.create_circle((a2_pos[0] + 0.5) * x_unit, (a2_pos[1] + 0.5) * y_unit,
                       x_unit * 0.5, a2_color)

  def _update_canvas_overlay(self):
    pass

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()


if __name__ == "__main__":
  app = BoxPushApp()
  app.run()
