import numpy as np
from models.utils.data_utils import Trajectories
from domains.cleanup_single.simulator import CleanupSingleSimulator
from domains.cleanup_single.mdp import MDPCleanupSingle


class CleanupSingleTrajectories(Trajectories):

  def __init__(self, task_mdp: MDPCleanupSingle) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=1,
                     num_latent_factors=1,
                     tup_num_latents=(task_mdp.num_latents, ))
    self.task_mdp = task_mdp

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = CleanupSingleSimulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        bstate, a_pos, a_act, a_lat = vec_state_action

        sidx = self.task_mdp.conv_sim_states_to_mdp_sidx([bstate, a_pos])
        aidx1 = (a_act if a_act is not None else Trajectories.EPISODE_END)

        xidx1 = (self.task_mdp.latent_space.state_to_idx[a_lat]
                 if a_lat is not None else Trajectories.EPISODE_END)

        np_trj[tidx, :] = [sidx, aidx1, xidx1]

      self.list_np_trajectory.append(np_trj)
