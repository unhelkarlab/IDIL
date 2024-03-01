import numpy as np
from domains.box_push.simulator import BoxPushSimulator
from models.utils.data_utils import Trajectories
from domains.box_push.mdp import BoxPushTeamMDP, BoxPushMDP


class BoxPushTrajectories(Trajectories):

  def __init__(self, task_mdp: BoxPushTeamMDP, agent_mdp: BoxPushMDP) -> None:
    super().__init__(num_state_factors=1,
                     num_action_factors=2,
                     num_latent_factors=2,
                     tup_num_latents=(agent_mdp.num_latents,
                                      agent_mdp.num_latents))
    self.task_mdp = task_mdp
    self.agent_mdp = agent_mdp

  def load_from_files(self, file_names):
    for file_nm in file_names:
      trj = BoxPushSimulator.read_file(file_nm)
      if len(trj) == 0:
        continue

      np_trj = np.zeros((len(trj), self.get_width()), dtype=np.int32)
      for tidx, vec_state_action in enumerate(trj):
        bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = vec_state_action

        sidx = self.task_mdp.conv_sim_states_to_mdp_sidx([bstt, a1pos, a2pos])
        aidx1 = (a1act if a1act is not None else Trajectories.EPISODE_END)
        aidx2 = (a2act if a2act is not None else Trajectories.EPISODE_END)

        xidx1 = (self.agent_mdp.latent_space.state_to_idx[a1lat]
                 if a1lat is not None else Trajectories.EPISODE_END)
        xidx2 = (self.agent_mdp.latent_space.state_to_idx[a2lat]
                 if a2lat is not None else Trajectories.EPISODE_END)

        np_trj[tidx, :] = [sidx, aidx1, aidx2, xidx1, xidx2]

      self.list_np_trajectory.append(np_trj)
