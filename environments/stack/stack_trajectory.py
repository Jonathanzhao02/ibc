from ibc.environments.stack.stack import StackEnv
from gym.envs import registration
from gym.spaces import Dict, Box

import numpy as np
import gin

from abr_control.controllers import Damping # type: ignore
from ibc.environments.utils.mujoco.my_osc import OSC

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

handle = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

bert_preprocess_model = hub.KerasLayer(preprocess)
bert_model = hub.KerasLayer(handle)

@gin.configurable
class StackTrajectoryEnv(StackEnv):
  @gin.configurable
  def __init__(self, dist_tolerance=0.01, encode_text=True, **kwargs):
    if encode_text:
        observation_space = Dict({
          "image": Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
          "objective": Box(low=0, high=255, shape=(100,), dtype='B'),
          "encoded_objective": Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
        })

        self.encoded_objective = None
    else:
      observation_space = Dict({
        "image": Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
        "objective": Box(low=0, high=255, shape=(100,), dtype='B'),
      })

    self.dist_tolerance = dist_tolerance
    self.encode_text = encode_text

    StackEnv.__init__(self, observation_space=observation_space, **kwargs)
  
  def reset_model(self):
    ob = super().reset_model()

    if self.encode_text:
      self.encoded_objective = bert_model(
        bert_preprocess_model([
          self.objective,
        ])
      )['pooled_output'].numpy().flatten()
      ob['encoded_objective'] = self.encoded_objective

    # Controller creation
    self.damping = Damping(self.robot_config, kv=10)
    self.controller = OSC(
      self.robot_config,
      kp=200,
      null_controllers=[self.damping],
      vmax=[0.5, 0.5],  # [m/s, rad/s]
      # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
      ctrlr_dof=[True, True, True, True, True, True],
      orientation_algorithm=1,
    )

    return ob
  
  # Action: [x, y, z, roll, pitch, yaw, gripper force]
  def step(self, a):
    feedback = self.mujoco_interface.get_feedback()
    u = self.controller.generate(
      q=feedback['q'],
      dq=feedback['dq'],
      target=a[:-1],
    )
    u[-1] = a[-1]
    ob, reward, terminated, _ = StackEnv.step(self, u)

    if self.encode_text:
      ob['encoded_objective'] = self.encoded_objective

    return ob, reward, terminated, {}

registration.register(id='StackTrajectory-v0', entry_point=StackTrajectoryEnv)
