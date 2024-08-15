import gymnasium as gym
import numpy as np 

class FrankaKitchenWrapper(gym.Wrapper):
    """
    Wrapper class for the FrankaKitchen environment from gymnasium. Purpose is to
    make it compliant with the legacy training code from original IDIL implementation.
    """
    def __init__(self, env):
        super(FrankaKitchenWrapper, self).__init__(env)
        # TODO: create latent data structures

    def reset(self):
        obs, completion = self.env.reset()
        return obs
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # NOTE: no support for latent handling is needed 
        # because the input is only the agent's action

        return obs, reward, terminated, info