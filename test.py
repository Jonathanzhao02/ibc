from ibc.environments.stack import stack
from ibc.environments.stack import stack_trajectory
from ibc.environments.stack_lite import stack_lite
from ibc.environments.stack_lite import stack_lite_trajectory
import gym

from PIL import Image

import matplotlib.pyplot as plt

import random

env = gym.make('StackLiteTrajectory-v0', render_mode='rgb_array')
env.reset()

# for _ in range(10):
#     for _ in range(20):
#         obs, _, _, _ = env.step([0.3, 0.3, 0.3, -1.57, 0, -1.57, -0.12])
    
#     env.reset()

# '''
# Visualize demos
# '''
# import tensorflow as tf
# from tf_agents.utils import example_encoding_dataset as ex

# from pathlib import Path

# # records = Path('ibc/data/stack').glob('*.tfrecord')
# # records = ['ibc/data/stack/stack_0.tfrecord']
# # t = ex.load_tfrecord_dataset([str(p) for p in records], compress_image=True)

# # for i,data in enumerate(iter(t)):
# #   d = data.observation['image']
# #   img = tf.keras.utils.array_to_img(tf.squeeze(d))
# #   img.save(f'/tmp/ibc/images/stack/{i}.png')

# records = Path('ibc/data/stack_trajectory').glob('*.tfrecord')
# t = ex.load_tfrecord_dataset([str(p) for p in records], compress_image=True)

# for i,data in enumerate(iter(t)):
#   d = data.observation['image']
#   img = tf.keras.utils.array_to_img(tf.squeeze(d))
#   img.save(f'/tmp/ibc/images/stack_trajectory/{i}.png')