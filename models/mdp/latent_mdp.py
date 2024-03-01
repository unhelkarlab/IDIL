import abc
import numpy as np
from tqdm import tqdm
import logging

from models.mdp import MDP
from models.utils.mdp_utils import StateSpace


class LatentMDP(MDP):
  """MDP with one latent state"""

  def __init__(self, fast_cache_mode: bool = False, use_sparse: bool = False):
    super().__init__(fast_cache_mode, use_sparse)

    # Define latent state space.
    self.init_latentspace()
    self.init_latentspace_helper_vars()

  @abc.abstractmethod
  def init_latentspace(self):
    """Defines MDP latent state space. """
    self.latent_space = StateSpace()

  def init_latentspace_helper_vars(self):
    """Creates helper variables for the latent state space."""
    self.num_latents = self.latent_space.num_states
    logging.debug("num_latents= %d" % (self.num_latents, ))

  @abc.abstractmethod
  def reward(self, latent_idx: int, state_idx: int, action_idx: int, *args,
             **kwargs) -> float:
    """Defines MDP reward function.

      Args:
        latent_idx: Index of an MDP latent.
        state_idx: Index of an MDP state.
        action_idx: Index of an MDP action.

      Returns:
        A scalar reward.
    """
    raise NotImplementedError

  @property
  def np_reward_model(self):
    """Returns reward model as a np ndarray."""
    # This code is largely repetitive to the parent method but more readable

    # If already computed, return the computed value.
    # This model does not change after the MDP is defined.
    if self._np_reward_model is not None:
      return self._np_reward_model

    # Else: Compute using the reward method.
    # Set -inf to unreachable states to prevent an action from falling in them
    self._np_reward_model = np.full(
        (self.num_latents, self.num_states, self.num_actions), -np.inf)

    for latent in range(self.num_latents):
      pbar = tqdm(range(self.num_states))
      for state in pbar:
        pbar.set_postfix(
            {'Latent Count': str(latent + 1) + '/' + str(self.num_latents)})
        if self.is_terminal(state) or len(self.legal_actions(state)) == 0:
          self._np_reward_model[latent, state, 0] = 0
        else:
          for action in self.legal_actions(state):
            self._np_reward_model[latent, state,
                                  action] = self.reward(latent, state, action)
    return self._np_reward_model
