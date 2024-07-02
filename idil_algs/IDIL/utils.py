from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from collections import defaultdict
import numpy as np
import pickle as pkl
import torch
import os
import multiprocessing as mp
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())


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
      assert len(states[traj_idx]) == len(
          actions[traj_idx]) == len(latents[traj_idx])
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
    self.actions = self._flatten_list(
        self.expert_dataset.trajectories["actions"])
    self.latents = self._flatten_list(
        self.expert_dataset.trajectories["latents"])

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
      probs = [np.where(np.array(self.expert_policy[(s, l)]) ==
                        a, 1-eps, eps).mean() for l in latents]
      # mask nans with 0 when non existent
      probs = np.nan_to_num(probs, nan=eps)
      log_probs.append(np.log(probs))

    log_probs = np.array(log_probs)
    return torch.tensor(log_probs, device=self.device)

  def save(self, path, suffix=""):
    """
    Store pickle file with expert policy object
    """

    with open(path + f"expert_policy{suffix}.pkl", "wb") as f:
      pkl.dump(self.expert_policy, f)


class ContinuousExpertPolicySampler:
  """
  Expert policy sampler used for continuous action spaces.
  This creates a dataset of (state, latent, action) tuples and provides a 
  choose_action method to sample from a KDE of the nearest seen actions.
  """

  def __init__(self, dataset, device, bandwidth=0.1, n_neighbors=10):
    self.expert_dataset = dataset
    self.device = device
    self.bandwidth = bandwidth
    self.n_neighbors = n_neighbors
    self._create_expert_policy()

  def _create_expert_policy(self):
    """
    Parse state, actions and latent lists from dataset and create a list of (state, latent, action) tuples.
    Also create KNN models for efficient nearest neighbor search.
    """
    self.states = np.array(self._flatten_list(self.expert_dataset.trajectories["states"]))
    self.actions = np.array(self._flatten_list(self.expert_dataset.trajectories["actions"]))
    self.latents = np.array(self._flatten_list(self.expert_dataset.trajectories["latents"]))
    self.latent_space = set(self.latents) # compute unique latent values
    
    self.state_action_stack = np.hstack([self.states, self.actions])

    self.data = np.hstack([self.states, self.latents.reshape(-1, 1)])

    # Create KNN model for (state, latent) for action sampling
    self.knn_model = NearestNeighbors(
        n_neighbors=self.n_neighbors, algorithm='auto').fit(self.data)
    
    # Create KNN model for (state, action) for latent logprob 
    self.knn_model_sa = NearestNeighbors(
        n_neighbors=self.n_neighbors, algorithm='auto').fit(self.state_action_stack)

  def _flatten_list(self, nested_list):
    return [item for sublist in nested_list for item in sublist]


  # TODO: refactor this function because KNN.kneighbors function is fully 
  # vectorized
  def _get_knbr_actions(self, state, latent):
    """Aux function to query the KNN model for nearest neighbors"""
    query = np.hstack([state, latent]).reshape(1, -1)
    _, index = self.knn_model.kneighbors(query)
    return self.actions[index.squeeze()]

  def choose_action(self, state, latent, *args):
    """
    Sample an action from the expert policy given a state and latent using KNN and KDE.
    """
    nearest_actions = self._get_knbr_actions(state, latent)

    kde = KernelDensity(
        kernel='gaussian', bandwidth=self.bandwidth).fit(nearest_actions)
    sampled_action = kde.sample(1)[0]

    return sampled_action

  def iq_update(self, *args, **kwargs):
    """
    Mock update and return 0 loss as we're directly matching the expert policy
    """
    return 0
  
  def _compute_log_prob(self, state, action, latent):
    # get nearest actions
    nearest_actions = self._get_knbr_actions(state, latent)

    # compute kde
    kde = KernelDensity(
        kernel='gaussian',
        bandwidth=self.bandwidth).fit(nearest_actions)
    
    log_prob = 1 / (1 + np.exp(-kde.score_samples([action])[0]))  # convert log-likelihood to prob
    return np.log(log_prob)

  def _get_latent_log_prob(self, latent_array): 
    """
    Given an array of observed latents, compute the 
    discrete probability across the entire model's latent space
    """
    if not isinstance(latent_array, np.ndarray):
      latent_array = np.array(latent_array)

    # Get discrete probabilities for each latent (latents are discrete!)
    eps = 1e-8
    latent_probs = np.array([np.where(latent_array == l, 1-eps, eps).mean() 
                             for l in self.latent_space]) # ensure returning vector is (len_state_arr, latent_dim) shape
    log_probs = np.log(latent_probs)
    return log_probs

  def log_probs(self, state_array, action_array):
    """
    Given two arrays, of states and actions, compute the closest 
    state and action samples for each zipped pair, then build a KDE estimator
    to compute the log probability for each latent

    Output size is (len(state_array), len(self.latents))
    """
    stacked_sa = np.hstack([state_array, action_array])

    # get nearest nbrs matrix for (state, action) KNN model
    _, indices = self.knn_model_sa.kneighbors(stacked_sa)

    # get nearest intents for those actions
    latents_matrix = self.latents[indices]
    log_probs = []
    for nbr_latents in latents_matrix:
      log_probs.append(self._get_latent_log_prob(nbr_latents))
    
    log_probs = torch.tensor(log_probs, device=self.device)
    return log_probs


  def save(self, path, suffix=""):
    """
    Store pickle file with expert policy object
    """
    with open(path + f"expert_policy{suffix}.pkl", "wb") as f:
      pkl.dump((self.data, self.actions), f)
