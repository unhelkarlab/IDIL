'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
import os
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/juanhevia/.mujoco/mujoco210/bin" + ":/usr/lib/nvidia"
from gym.envs.registration import register
from gymnasium.envs.registration import register as register_gymnasium



register(id='envfrommdp-v0',
         entry_point='idil_gym.envs.mdp_envs:EnvFromMDP',
         max_episode_steps=200)
register(id='envfromcallbacks-v0',
         entry_point='idil_gym.envs.mdp_env:EnvFromCallbacks',
         max_episode_steps=200)
register(id='envfromlatentmdp-v0',
         entry_point='idil_gym.envs.mdp_env:EnvFromLatentMDP',
         max_episode_steps=200)
register(id='envaicoaching-v0',
         entry_point='idil_gym.envs.mdp_env:EnvFromLearnedModels',
         max_episode_steps=200)
register(id='envaicoachingnoop-v0',
         entry_point='idil_gym.envs.mdp_env:EnvFromLearnedModelsNoop',
         max_episode_steps=200)

register(id='circleworld-v0',
         entry_point='idil_gym.envs:CircleWorld',
         max_episode_steps=50)

for idx in range(1, 6):
  register(id=f'MultiGoals2D_{idx}-v0',
           entry_point=f'idil_gym.envs:MultiGoals2D_{idx}',
           max_episode_steps=200)

register(id='AntPush-v0',
         entry_point='idil_gym.envs.ant_maze_env_ex:AntPushEnv_v0',
         max_episode_steps=1000)

register(id='CleanupSingle-v0',
         entry_point='idil_gym.envs.cleanup_single:CleanupSingleEnv_v0',
         max_episode_steps=200)

register(id='EnvMovers-v0',
         entry_point='idil_gym.envs.box_push_for_two:EnvMovers_v0',
         max_episode_steps=200)

register(id='EnvCleanup-v0',
         entry_point='idil_gym.envs.box_push_for_two:EnvCleanup_v0',
         max_episode_steps=200)

register(id='RMPickPlaceCan-v0',
         entry_point='idil_gym.envs.robomimic_env:RMPickPlaceCan',
         max_episode_steps=400)

register_gymnasium(id='CustomFrankaKitchen_014-v0',
                   entry_point='idil_gym.envs.franka_kitchen:CustomFrankaKitchen_014',
                   max_episode_steps=280)

register_gymnasium(id='CustomFrankaKitchen_012-v0',
                   entry_point='idil_gym.envs.franka_kitchen:CustomFrankaKitchen_012',
                   max_episode_steps=280)

register_gymnasium(id='CustomFrankaKitchen_042-v0',
                    entry_point='idil_gym.envs.franka_kitchen:CustomFrankaKitchen_042',
                    max_episode_steps=280)

register_gymnasium(id='CustomFrankaKitchen_142-v0',
                    entry_point='idil_gym.envs.franka_kitchen:CustomFrankaKitchen_142',
                    max_episode_steps=280)