from typing import Hashable, Mapping, Tuple, Sequence
import os
import numpy as np
import random
from domains.simulator import Simulator
from domains.agent import SimulatorAgent, InteractiveAgent
from domains.box_push import (EventType, AGENT_ACTIONSPACE,
                              conv_box_idx_2_state, BoxState)
from .transition import transition_single_agent

Coord = Tuple[int, int]


class CleanupSingleSimulator(Simulator):
  AGENT_ID = 0

  def __init__(self, fix_init: bool = False) -> None:
    #  input1: agent idx
    super().__init__(0)
    self.agent = None
    self.fix_init = fix_init

  def init_game(self,
                x_grid: int,
                y_grid: int,
                init_pos: Coord,
                boxes: Sequence[Coord] = [],
                goals: Sequence[Coord] = [],
                walls: Sequence[Coord] = [],
                drops: Sequence[Coord] = [],
                wall_dir: Sequence[int] = [],
                **kwargs):
    self.x_grid = x_grid
    self.y_grid = y_grid
    self.init_pos = init_pos
    self.boxes = boxes
    self.goals = goals
    self.walls = walls
    self.drops = drops
    self.wall_dir = wall_dir

    self.possible_positions = []
    for x in range(x_grid):
      for y in range(y_grid):
        pos = (x, y)
        if pos not in walls and pos not in goals and pos not in boxes:
          self.possible_positions.append(pos)

    self.reset_game()

  def get_state(self):
    return [self.box_states, self.agent_pos]

  def set_autonomous_agent(self, agent: SimulatorAgent = InteractiveAgent()):
    self.agent = agent
    self.agent.init_latent(self.get_state())

  def reset_game(self):
    self.current_step = 0
    self.history = []

    if self.fix_init:
      self.agent_pos = self.init_pos
    else:
      self.agent_pos = random.choice(self.possible_positions)

    # starts with their original locations
    self.box_states = [0] * len(self.boxes)

    if self.agent is not None:
      self.agent.init_latent(self.get_state())
    self.changed_state = set()

  def get_score(self):
    return -self.get_current_step()

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    agent_action = map_agent_2_action[self.AGENT_ID]

    if agent_action is None:
      return

    agent_lat = self.agent.get_current_latent()
    if agent_lat is None:
      agent_lat = ("NA", 0)
    agent_lat_0 = agent_lat[0] if agent_lat[0] is not None else "NA"
    agent_lat_1 = agent_lat[1] if agent_lat[1] is not None else 0
    agent_lat = (agent_lat_0, agent_lat_1)

    cur_state = tuple(self.get_state())

    cur_info = [self.current_step, *cur_state, agent_action, agent_lat]

    self._transition(agent_action)
    self.current_step += 1
    self.changed_state.add("current_step")
    tuple_actions = (agent_action, )

    TEST_CODE = False
    if TEST_CODE:
      mdp = self.agent.agent_model.get_reference_mdp()
      sidx = mdp.conv_sim_states_to_mdp_sidx(cur_state)
      aidx = mdp.conv_sim_actions_to_mdp_aidx(tuple_actions)
      xidx = self.agent.conv_latent_to_idx(agent_lat)
      print(mdp.reward(xidx, sidx, aidx))

    # update mental model
    self.agent.update_mental_state(cur_state, tuple_actions, self.get_state())
    self.changed_state.add("agent_latent")

    cur_info.append(self.get_score())
    self.history.append(cur_info)

  def _transition(self, agent_action):
    list_next_env = self._get_transition_distribution(agent_action)

    list_prop = []
    for item in list_next_env:
      list_prop.append(item[0])

    idx_c = np.random.choice(range(len(list_next_env)), 1, p=list_prop)[0]
    _, box_states, agent_pos = list_next_env[idx_c]
    self.agent_pos = agent_pos
    self.box_states = box_states

    self.changed_state.add("agent_pos")
    self.changed_state.add("box_states")

  def _get_transition_distribution(self, agent_action):
    return transition_single_agent(self.box_states, self.agent_pos,
                                   agent_action, self.boxes, self.goals,
                                   self.walls, self.drops, self.x_grid,
                                   self.y_grid, self.init_pos)

  def get_num_agents(self):
    return 1

  def event_input(self, agent: Hashable, event_type: Hashable, value):
    if event_type is None:
      return

    if event_type != EventType.SET_LATENT:
      self.agent.set_action(event_type)
    else:
      self.agent.set_latent(value)
      self.changed_state.add("agent_latent")

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:
    map_a2a = {}
    map_a2a[self.AGENT_ID] = self.agent.get_action(self.get_state())
    return map_a2a

  def get_env_info(self):
    return {
        "x_grid": self.x_grid,
        "y_grid": self.y_grid,
        "box_states": self.box_states,
        "boxes": self.boxes,
        "goals": self.goals,
        "drops": self.drops,
        "walls": self.walls,
        "agent_pos": self.agent_pos,
        "agent_latent": self.agent.get_current_latent(),
        "wall_dir": self.wall_dir,
        "current_step": self.current_step
    }

  def get_changed_objects(self):
    dict_changed_obj = {}
    for state in self.changed_state:
      if state == "agent_latent":
        dict_changed_obj[state] = self.agent.get_current_latent()
      else:
        dict_changed_obj[state] = getattr(self, state)
    self.changed_state = set()
    return dict_changed_obj

  def save_history(self, file_name, header):
    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
      # sequence
      txtfile.write(header)
      txtfile.write('\n')
      txtfile.write('# cur_step, box_state, agent_pos, agent_act, agent_latent')
      txtfile.write('\n')

      for step, bstt, a_pos, a_act, a_lat, scr in self.history:
        txtfile.write('%d; ' % (step, ))  # cur step
        # box states
        for idx in range(len(bstt) - 1):
          txtfile.write('%d, ' % (bstt[idx], ))
        txtfile.write('%d; ' % (bstt[-1], ))

        txtfile.write('%d, %d; ' % a_pos)

        txtfile.write('%d; ' % (AGENT_ACTIONSPACE.action_to_idx[a_act], ))

        txtfile.write('%s, %d; ' % a_lat)
        txtfile.write('\n')

      # last state
      txtfile.write('%d; ' % (self.current_step, ))  # cur step
      # box states
      for idx in range(len(self.box_states) - 1):
        txtfile.write('%d, ' % (self.box_states[idx], ))
      txtfile.write('%d; ' % (self.box_states[-1], ))

      txtfile.write('%d, %d; ' % self.agent_pos)
      txtfile.write('\n')

  def is_finished(self) -> bool:
    if super().is_finished():
      return True

    return self.check_task_done()

  def check_task_done(self):
    n_drops = len(self.drops)

    for state in self.box_states:
      if conv_box_idx_2_state(state, n_drops)[0] != BoxState.OnGoalLoc:
        return False
    return True

  @classmethod
  def read_file(cls, file_name):
    traj = []
    with open(file_name, newline='') as txtfile:
      lines = txtfile.readlines()
      i_start = 0
      for i_r, row in enumerate(lines):
        if row == '# cur_step, box_state, agent_pos, agent_act, agent_latent\n':
          i_start = i_r
          break

      for i_r in range(i_start + 1, len(lines)):
        line = lines[i_r]
        states = line.rstrip()[:-1].split("; ")
        if len(states) < 5:
          for dummy in range(5 - len(states)):
            states.append(None)
        step, bstate, a_pos, a_act, a_lat = states
        box_state = tuple([int(elem) for elem in bstate.split(", ")])
        agent_pos = tuple([int(elem) for elem in a_pos.split(", ")])
        if a_act is None:
          agent_act = None
        else:
          agent_act = int(a_act)
        if a_lat is None:
          agent_lat = None
        else:
          a_lat_tmp = a_lat.split(", ")
          agent_lat = (a_lat_tmp[0], int(a_lat_tmp[1]))
        traj.append([box_state, agent_pos, agent_act, agent_lat])

    return traj
