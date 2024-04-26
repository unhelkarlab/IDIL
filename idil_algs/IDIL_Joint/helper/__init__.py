'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''

from idil_algs.baselines.IQLearn.utils.logger import AGENT_TRAIN_FORMAT

AGENT_TRAIN_FORMAT['oiql'] = [
    # ('batch_reward', 'BR', 'float'),
    ('actor_loss', 'ALOSS', 'float'),
    ('critic_loss', 'CLOSS', 'float'),
    ('alpha_loss', 'TLOSS', 'float'),
    ('alpha_value', 'TVAL', 'float'),
    ('actor_entropy', 'AENT', 'float')
]

AGENT_TRAIN_FORMAT['osac'] = [
    # ('batch_reward', 'BR', 'float'),
    ('actor_loss', 'ALOSS', 'float'),
    ('critic_loss', 'CLOSS', 'float'),
    ('alpha_loss', 'TLOSS', 'float'),
    ('alpha_value', 'TVAL', 'float'),
    ('actor_entropy', 'AENT', 'float')
]
