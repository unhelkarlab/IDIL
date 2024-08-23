# import gymnasium as gym
from gymnasium import Wrapper
import gymnasium
import numpy as np

class FrankaKitchenWrapper(Wrapper):
    """
    Wrapper class for the FrankaKitchen environment from gymnasium. Purpose is to
    make it compliant with the legacy training code from original IDIL implementation.
    """

    metadata = {'render.modes': ['human']}
    LATENT_IDX_MAPPING = {
            "microwave": 0,
            "kettle": 1,
            "light switch": 2,
            "slide cabinet": 3,
            "bottom burner": 4,
            "top burner": 5,
            "hinge cabinet": 6
        }
    IDX_LATENT_MAPPING = {v:k for k,v in LATENT_IDX_MAPPING.items()}    

    def __init__(self,
                 macro_goals : list = [1, 3, 5],
                 seed: int = None): # default macro goals based on the trajectories availability

        goals = [self.IDX_LATENT_MAPPING[goal] for goal in macro_goals]

        if not seed:
            seed = np.random.randint(0, 1000)

        env = gymnasium.make("FrankaKitchen-v1", tasks_to_complete=goals,
                              apply_api_compatibility=False,
                              disable_env_checker=True,)
        super().__init__(env)

    def reset(self, seed: int = None):
        if not seed:
            seed = np.random.randint(0, 1000)
        self.env.reset()
        state, completion = self.unwrapped.reset(seed=seed)
        return {
            'state': state,
            'completion': completion
            }

    def step(self, action):
        state_obj, reward, terminated, truncated, info = self.env.step(action)

        return state_obj, reward, terminated, truncated, info
    
# create particular environments for each specific trajectory
class CustomFrankaKitchen_014(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[0,1,4])

class CustomFrankaKitchen_012(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[0,1,2])

class CustomFrankaKitchen_042(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[0,4,2])

class CustomFrankaKitchen_142(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[1,4,2])