from typing import Sequence, Optional
import gym
from gym import spaces
from models.mdp import MDP
import numpy as np


class EnvFromMDP(gym.Env):
  # uncomment below line if you need to render the environment
  # metadata = {'render.modes': ['console']}

  def __init__(self,
               mdp: MDP,
               possible_init_states: Optional[Sequence[int]] = None,
               init_state_dist: Optional[np.ndarray] = None,
               use_central_action=False):
    '''
    either possible_init_state or init_state_dist should not be None
    '''
    assert possible_init_states or init_state_dist

    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.use_central_action = use_central_action

    if mdp.num_action_factors == 1 or use_central_action:
      self.action_space = spaces.Discrete(mdp.num_actions)
    else:
      self.action_space = spaces.MultiDiscrete(mdp.list_num_actions)

    self.observation_space = spaces.Discrete(mdp.num_states)

    self.mdp = mdp
    self.possible_init_states = possible_init_states
    self.init_state_dist = init_state_dist

    if possible_init_states:
      self.sample = lambda: int(np.random.choice(self.possible_init_states))
    else:
      self.sample = lambda: int(
          np.random.choice(mdp.num_states, p=self.init_state_dist))

    self.cur_state = self.sample()

  def step(self, action):
    info = {}
    if self.use_central_action or len(self.mdp.list_num_actions) == 1:
      action_idx = action
    else:
      action_idx = self.mdp.conv_action_to_idx(tuple(action))
    # action_idx = (action if len(self.mdp.list_num_actions) == 1 else
    #               self.mdp.conv_action_to_idx(tuple(action)))

    if action_idx not in self.mdp.legal_actions(self.cur_state):
      info["invalid_transition"] = True
      return self.cur_state, -10000, False, info

    self.cur_state = self.mdp.transition(self.cur_state, action_idx)

    reward = -1  # we don't need reward for imitation learning
    done = self.mdp.is_terminal(self.cur_state)

    return self.cur_state, reward, done, info

  def reset(self):
    self.cur_state = self.sample()

    return self.cur_state  # reward, done, info can't be included

  # implement render function if need to be
  # def render(self, mode='human'):
  #   ...

  # implement close function if need to be
  # def close(self):
  #   ...
