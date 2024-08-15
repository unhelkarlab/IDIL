'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

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

# TODO: add register for FrankaKitchen
register(id='FrankaKitchen-v0',
         entry_point='idil_gym.envs.franka_kitchen:FrankaKitchen',  )