from collections import defaultdict
import numpy as np
import pickle as pkl
import torch
import os

class DiscreteExpertPolicySampler:
    """
    Expert policy sampler used for discrete action spaces only.
    This creates a dictionary where keys are (state, latent) pairs and provides a 
    choose_action method to sample uniformly from seen actions in that state-latent pair.
    """
    def __init__(self, dataset, device):
        self.expert_dataset = dataset
        self._create_expert_policy()
        self.device = device

    def _create_expert_policy(self):
        """
        Parse state, actions and latent lists from dataset and create a dictionary where 
        keys are (state, latent) pairs and values are lists of actions taken in that state-latent pair.
        """
        states = self.expert_dataset.trajectories["states"]
        actions = self.expert_dataset.trajectories["actions"]
        latents = self.expert_dataset.trajectories["latents"]

        state_latent_actions = defaultdict(list)
        state_actions = defaultdict(list)
        latent_actions = defaultdict(list)
        
        for traj_idx in range(len(states)):
            assert len(states[traj_idx]) == len(actions[traj_idx]) == len(latents[traj_idx])
            _states_idx = states[traj_idx]
            _actions_idx = actions[traj_idx]
            _latents_idx = latents[traj_idx]

            for state, action, latent in zip(_states_idx, _actions_idx, _latents_idx):
                state_latent_actions[(state, latent)].append(action)
                state_actions[state].append(action)
                latent_actions[latent].append(action)

        self.expert_policy = state_latent_actions
        self.state_action_marg = state_actions
        self.latent_action_marg = latent_actions
        self.state_latent_pairs = list(state_latent_actions.keys())
        
        # store flattened latents, states and actions
        self.states = self._flatten_list(self.expert_dataset.trajectories["states"])
        self.actions = self._flatten_list(self.expert_dataset.trajectories["actions"])
        self.latents = self._flatten_list(self.expert_dataset.trajectories["latents"])

    def _flatten_list(self, nested_list):
        return [item for sublist in nested_list for item in sublist]
    
    def choose_action(self, state, latent, *args):
        """
        Sample an action from the expert policy given a state and latent.
        """
        seen_state = state in self.states
        seen_latent = latent in self.latents
        seen_key = (state, latent) in self.state_latent_pairs

        if seen_key:
            return np.random.choice(self.expert_policy[(state, latent)])
        elif seen_state:
            # sample uniformly from all actions in the state
            return np.random.choice(self.state_action_marg[state])
        elif seen_latent:
            # sample uniformly from all actions in the latent
            return np.random.choice(self.latent_action_marg[latent])
        # it state and latent are new, sample uniformly from all actions
        return np.random.choice(self.actions)

    def iq_update(self, *args, **kwargs):
        """
        Mock update and return 0 loss as we're directly matching the expert
        policy
        """
        return 0
    
    def log_probs(self, state, action):
        """
        Given a list of states and actions belonging to a trajectory,
        compute the log probabilities of each latent at each (state, action) pair.
        """
        log_probs = []
        latents = set(self.latents)
        eps = 1e-5
        for s, a in zip(state, action):
            # compute probs for each action in the (s, l) pair
            probs = [np.where(np.array(self.expert_policy[(s, l)]) == a, 1-eps, eps).mean() for l in latents]
            # mask nans with 0 when non existent
            probs = np.nan_to_num(probs, nan=eps)
            log_probs.append(np.log(probs))

        log_probs = np.array(log_probs)
        return torch.tensor(log_probs, device=self.device)

    def save(self, path, suffix=""):
        """
        Store pickle file with expert policy object
        """

        with open(path+ f"expert_policy{suffix}.pkl", "wb") as f:
            pkl.dump(self.expert_policy, f)