from ibc.environments.stack_lite.stack_lite import StackLiteEnv
from gym.envs import registration
from gym.spaces import Dict, Box

import numpy as np
import gin

from abr_control.controllers import Damping # type: ignore
from ibc.environments.utils.mujoco.my_osc import OSC
import ibc.environments.stack.stack_viz as stack_viz

import os
import matplotlib.pyplot as plt

@gin.configurable
class StackLiteTrajectoryEnv(StackLiteEnv):
  @gin.configurable
  def __init__(self, xml_path='ibc/environments/assets/my_models/ur5_robotiq85/ur5_tabletop_static.xml', dist_tolerance=0.01, random_place=True, **kwargs):
    observation_space = Dict({
      "rgb": Box(low=0, high=255, shape=(224,224,3), dtype=np.float32),
    })

    self.dist_tolerance = dist_tolerance
    self.goal_log = []

    StackLiteEnv.__init__(self, xml_path=xml_path, observation_space=observation_space, random_place=random_place, **kwargs)
  
  def reset_model(self):
    ob = super().reset_model()
    self.goal_log = []

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
    self.goal_log.append(a[:3])
    ob, reward, terminated, _ = StackLiteEnv.step(self, u)

    return ob, reward, terminated, {}

  def save_image(self, traj):
    if traj.is_last():
      assert self.img_save_dir is not None  # pytype: disable=attribute-error
      fig, _ = stack_viz.visualize(self.pos_log, self.goal_log)
      filename = os.path.join(self.img_save_dir,  # pytype: disable=attribute-error
                              str(self.reset_counter).zfill(6)+'_2d.png')
      fig.savefig(filename)
      plt.close(fig)

registration.register(id='StackLiteTrajectory-v0', entry_point=StackLiteTrajectoryEnv)
