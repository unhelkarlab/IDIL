from train import load_expert_data_w_labels
from collections import defaultdict
import numpy as np

class DiscreteExpertPolicySampler:
    """
    Expert policy sampler used for discrete action spaces only.
    This creates a dictionary where keys are (state, latent) pairs and provides a 
    choose_action method to sample uniformly from seen actions in that state-latent pair.
    """
    def __init__(self, dataset):
        self.expert_dataset = dataset
        self.expert_policy = self._create_expert_policy()

    def _create_expert_policy(self):
        """
        Parse state, actions and latent lists from dataset and create a dictionary where 
        keys are (state, latent) pairs and values are lists of actions taken in that state-latent pair.
        """
        states = self.expert_dataset.trajectories["states"]
        actions = self.expert_dataset.trajectories["actions"]
        latents = self.expert_dataset.trajectories["latents"]

        state_latent_actions = defaultdict(list)
        for traj_idx in range(len(states)):
            assert len(states[traj_idx]) == len(actions[traj_idx]) == len(latents[traj_idx])
            _states_idx = states[traj_idx]
            _actions_idx = actions[traj_idx]
            _latents_idx = latents[traj_idx]

            for state, action, latent in zip(_states_idx, _actions_idx, _latents_idx):
                state_latent_actions[(state, latent)].append(action)

        return state_latent_actions
    
    def choose_action(self, state, latent, **kwargs):
        """
        Sample an action from the expert policy given a state and latent.
        """
        return np.random.choice(self.expert_policy[(state, latent)])
    

    def iq_update(self, *args, **kwargs):
        pass

