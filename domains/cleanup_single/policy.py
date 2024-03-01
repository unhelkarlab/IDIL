import os
from models.policy import CachedPolicyInterface
from domains.cleanup_single.mdp import MDPCleanupSingle

policy_cleanupsingle_list = []


class Policy_CleanupSingle(CachedPolicyInterface):

  def __init__(self, mdp: MDPCleanupSingle, temperature: float) -> None:
    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_cleanupsingle_")
    str_fileprefix += mdp.map_to_str() + "_"
    super().__init__(mdp, str_fileprefix, policy_cleanupsingle_list,
                     temperature, (0, ))
