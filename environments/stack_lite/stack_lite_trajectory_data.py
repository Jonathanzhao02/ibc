from ibc.environments.stack_lite.stack_lite_trajectory import StackLiteTrajectoryEnv
from gym.envs import registration
from gym.spaces import Dict, Box

import numpy as np
import gin

from abr_control.utils import transformations # type: ignore

@gin.configurable
class StackLiteTrajectoryDataEnv(StackLiteTrajectoryEnv):
  @gin.configurable
  def __init__(self, xml_path='ibc/environments/assets/my_models/ur5_robotiq85/ur5_tabletop_static.xml', dist_tolerance=0.01, random_place=True, **kwargs):
    observation_space = Dict({
      "rgb": Box(low=0, high=255, shape=(224,224,3), dtype=np.float32),
      "ee": Box(shape=(6,), dtype=np.float32),
    })

    self.dist_tolerance = dist_tolerance
    self.goal_log = []

    StackLiteTrajectoryEnv.__init__(self, xml_path=xml_path, observation_space=observation_space, random_place=random_place, **kwargs)
  
  def step(self, a):
    ob, reward, terminated, _ = StackLiteTrajectoryEnv.step(self, a)
    rot = transformations.euler_from_quaternion(self.mujoco_interface.get_orientation('EE'), "rxyz")
    pos = self.mujoco_interface.get_xyz('EE')
    ob['ee'] = np.concatenate((pos, rot))
    return ob, reward, terminated, {}

registration.register(id='StackLiteTrajectoryData-v0', entry_point=StackLiteTrajectoryDataEnv)
