from idil_gym.envs.mdp_env.env_from_mdp import EnvFromMDP
from hcair_domains.cleanup_single.mdp import MDPCleanupSingle
from hcair_domains.cleanup_single.maps import MAP_SINGLE_V1


class CleanupSingleEnv_v0(EnvFromMDP):

  def __init__(self):
    game_map = MAP_SINGLE_V1
    mdp = MDPCleanupSingle(**game_map)
    init_bstate = [0] * len(game_map["boxes"])
    init_pos = game_map["init_pos"]
    mdp.walls
    mdp.x_grid
    mdp.y_grid

    possible_init_states = []
    for x in range(mdp.x_grid):
      for y in range(mdp.y_grid):
        pt = (x, y)
        if pt not in mdp.walls and pt not in mdp.goals and pt not in mdp.boxes:
          sidx = mdp.conv_sim_states_to_mdp_sidx((init_bstate, pt))
          possible_init_states.append(sidx)

    # init_sidx = mdp.conv_sim_states_to_mdp_sidx((init_bstate, init_pos))
    # possible_init_states = [init_sidx]

    super().__init__(mdp, possible_init_states, use_central_action=True)


if __name__ == "__main__":
  from hcair_domains.cleanup_single.simulator import CleanupSingleSimulator
  from hcair_domains.cleanup_single.maps import MAP_SINGLE_V1
  from hcair_domains.cleanup_single.policy import Policy_CleanupSingle
  from hcair_domains.cleanup_single.mdp import MDPCleanupSingle
  from hcair_domains.cleanup_single.agent import Agent_CleanupSingle
  from collections import defaultdict
  import os
  import pickle

  sim = CleanupSingleSimulator()
  TEMPERATURE = 0.3
  GAME_MAP = MAP_SINGLE_V1
  mdp_task = MDPCleanupSingle(**GAME_MAP)
  policy = Policy_CleanupSingle(mdp_task, TEMPERATURE)
  agent = Agent_CleanupSingle(policy)

  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(agent)

  print(mdp_task.num_latents)
  print(mdp_task.num_states)

  # generate data
  ############################################################################
  if False:
    num_data = 50
    cur_dir = os.path.dirname(__file__)
    expert_trajs = defaultdict(list)
    for idx in range(num_data):
      while not sim.is_finished():
        map_agent_2_action = sim.get_joint_action()
        sim.take_a_step(map_agent_2_action)

      s_array = []
      a_array = []
      r_array = []
      x_array = []
      for _, bstt, a_pos, a_act, a_lat, scr in sim.history:
        sidx = mdp_task.conv_sim_states_to_mdp_sidx((bstt, a_pos))
        aidx = mdp_task.conv_sim_actions_to_mdp_aidx((a_act, ))
        xidx = agent.conv_latent_to_idx(a_lat)
        s_array.append(sidx)
        a_array.append(aidx)
        r_array.append(-1)
        x_array.append(xidx)

      sidx = mdp_task.conv_sim_states_to_mdp_sidx(
          (sim.box_states, sim.agent_pos))
      s_array.append(sidx)

      length = len(r_array)
      dones = [False] * length
      dones[-1] = sim.check_task_done()

      sim.reset_game()

      expert_trajs["states"].append(s_array[:-1])
      expert_trajs["next_states"].append(s_array[1:])
      expert_trajs["actions"].append(a_array)
      expert_trajs["latents"].append(x_array)
      expert_trajs["rewards"].append(r_array)
      expert_trajs["dones"].append(dones)
      expert_trajs["lengths"].append(length)

    file_path = os.path.join(cur_dir, f"CleanupSingle-v0_{num_data}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(expert_trajs, f)
