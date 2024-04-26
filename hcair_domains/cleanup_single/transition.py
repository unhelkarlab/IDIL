from typing import Tuple, Sequence
from hcair_domains.box_push import (BoxState, EventType, conv_box_state_2_idx,
                                    conv_box_idx_2_state)
from hcair_domains.box_push.transition import (hold_state_impl, is_wall_impl,
                                               update_dropped_box_state_impl)

Coord = Tuple[int, int]


def transition_single_agent(box_states: list, agent_pos: Coord,
                            agent_act: EventType,
                            box_locations: Sequence[Coord],
                            goals: Sequence[Coord], walls: Sequence[Coord],
                            drops: Sequence[Coord], x_bound: int, y_bound: int,
                            init_pos: Coord):
  num_drops = len(drops)
  num_goals = len(goals)

  # methods
  def get_box_idx(coord, box_states):
    for idx, sidx in enumerate(box_states):
      state = conv_box_idx_2_state(sidx, len(drops))
      if (state[0] == BoxState.Original) and (coord == box_locations[idx]):
        return idx
      elif (state[0] == BoxState.WithAgent1) and (coord == agent_pos):
        return idx
      elif (state[0] == BoxState.OnDropLoc) and (coord == drops[state[1]]):
        return idx

    return -1

  def possible_positions(coord, box_states, holding_box):
    x, y = coord
    possible_pos = []
    possible_pos.append(coord)

    for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      pos = (x + i, y + j)

      if not holding_box:
        if not is_wall_impl(pos, x_bound, y_bound, walls) and pos not in goals:
          possible_pos.append(pos)
      else:
        if (not is_wall_impl(pos, x_bound, y_bound, walls)
            and get_box_idx(pos, box_states) < 0):
          possible_pos.append(pos)

    return possible_pos

  def get_moved_coord(coord, action, box_states=None, holding_box=False):
    x, y = coord
    coord_new = None
    if action == EventType.UP:
      coord_new = (x, y - 1)
    elif action == EventType.DOWN:
      coord_new = (x, y + 1)
    elif action == EventType.LEFT:
      coord_new = (x - 1, y)
    elif action == EventType.RIGHT:
      coord_new = (x + 1, y)
    else:
      coord_new = coord

    if is_wall_impl(coord_new, x_bound, y_bound, walls):
      coord_new = coord

    if box_states is not None and get_box_idx(coord_new, box_states) >= 0:
      coord_new = coord

    if coord_new in goals and not holding_box:
      return coord
    else:
      return coord_new

  def get_dist_new_coord(coord, action, box_states=None, holding_box=False):
    list_possible_pos = possible_positions(coord, box_states, holding_box)
    expected_pos = get_moved_coord(coord, action, box_states, holding_box)
    P_EXPECTED = 0.95
    list_dist = []
    for pos in list_possible_pos:
      if pos == expected_pos:
        list_dist.append((P_EXPECTED, pos))
      else:
        p = (1 - P_EXPECTED) / (len(list_possible_pos) - 1)
        list_dist.append((p, pos))
    return list_dist

  def hold_state():
    return hold_state_impl(box_states, drops, goals)

  def update_dropped_box_state(boxidx, coord, box_states_new):
    res = update_dropped_box_state_impl(boxidx, coord, box_states_new,
                                        box_locations, drops, goals)
    return res, conv_box_idx_2_state(box_states_new[boxidx], num_drops,
                                     num_goals)

  list_next_env = []
  hold = hold_state()
  # both do not hold anything
  if hold == "None":
    if agent_act == EventType.HOLD:
      bidx = get_box_idx(agent_pos, box_states)
      if bidx >= 0:
        state1 = (BoxState.WithAgent1, None)
        box_states_new = list(box_states)
        box_states_new[bidx] = conv_box_state_2_idx(state1, num_drops)
        list_next_env.append((1.0, box_states_new, agent_pos))
      else:
        list_next_env.append((1.0, box_states, agent_pos))

    else:
      agent_pos_dist = [(1.0, agent_pos)]
      box_states_new = list(box_states)
      agent_pos_dist = get_dist_new_coord(agent_pos, agent_act, None, False)

      for p1, pos1 in agent_pos_dist:
        list_next_env.append((p1, box_states_new, pos1))
  # only a1 holds a box
  elif hold == "A1":
    box_states_new = list(box_states)
    a1_dropped = False
    agent_pos_dist = [(1.0, agent_pos)]
    if agent_act == EventType.UNHOLD:
      bidx = get_box_idx(agent_pos, box_states)
      assert bidx >= 0
      a1_dropped, bstate = update_dropped_box_state(bidx, agent_pos,
                                                    box_states_new)
      # respawn
      if bstate[0] == BoxState.OnGoalLoc:
        agent_pos_dist = [(1.0, init_pos)]

    if not a1_dropped:
      agent_pos_dist = get_dist_new_coord(agent_pos, agent_act, box_states_new,
                                          True)

    for p1, pos1 in agent_pos_dist:
      list_next_env.append((p1, box_states_new, pos1))
  else:  # invalid
    raise ValueError("invalid state")

  return list_next_env
