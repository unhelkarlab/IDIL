# import gymnasium as gym
from gymnasium import Wrapper
import gymnasium

class FrankaKitchenWrapper(Wrapper):
    """
    Wrapper class for the FrankaKitchen environment from gymnasium. Purpose is to
    make it compliant with the legacy training code from original IDIL implementation.
    """

    metadata = {'render.modes': ['human']}
    LATENT_IDX_MAPPING = {
            "microwave": 1,
            "kettle": 2,
            "light switch": 3,
            "slide cabinet": 4,
            "bottom burner": 5,
            "top burner": 6,
            "hinge cabinet": 7
        }
    
    IDX_LATENT_MAPPING = {
        1: "microwave",
        2: "kettle",
        3: "light switch",
        4: "slide cabinet",
        5: "bottom burner",
        6: "top burner",
        7: "hinge cabinet"
    }

    def __init__(self,
                 macro_goals : list = [1, 3, 5]): # default macro goals based on the trajectories availability

        goals = [self.IDX_LATENT_MAPPING[goal] for goal in macro_goals]

        env = gymnasium.make("FrankaKitchen-v1", tasks_to_complete=goals, apply_api_compatibility=True)
        super().__init__(env)

    def reset(self):
        state, completion = self.env.reset()
        return {
            'state': state,
            'completion': completion
        }

    def step(self, action):
        return self.env.step(action)

    # def reset(self):
    #     obs, _ = self.env.reset()
    #     return obs
        
    # def step(self, action):
    #     obs, reward, terminated, truncated, info = self.env.step(action)

    #     return obs, reward, terminated, info
    
# create particular environments for each specific trajectory
class CustomFrankaKitchen_125(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[1,2,5])

class CustomFrankaKitchen_123(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[1,2,3])

class CustomFrankaKitchen_153(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[1,5,3])

class CustomFrankaKitchen_253(FrankaKitchenWrapper):
    def __init__(self):
        super().__init__(macro_goals=[2,5,3])