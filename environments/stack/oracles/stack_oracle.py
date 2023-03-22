from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from ibc.environments.utils.mujoco.sequential_actions_interface import Executor
from ibc.environments.utils.mujoco.tasks import stack

from abr_control.controllers import Damping # type: ignore
from ibc.environments.utils.mujoco.my_osc import OSC

import numpy as np

MUG_PICKUP_DX = 0.04
MUG_PICKUP_DZ = 0.065
BOWL_PICKUP_DX = 0.08
BOWL_PICKUP_DZ = 0.042

MUG_PLACE_DZ = 0.2
BOWL_PLACE_DZ = 0.2
PLATE_PLACE_DZ = 0.1

DIST_MAX = 0.75
AVG_CHANGE = 0.15

class StackOracle(py_policy.PyPolicy):
  def __init__(self, env):
    super(StackOracle, self).__init__(env.time_step_spec(), env.action_spec())
    self._env = env
    self._gripper = None
  
  def reset(self):
    env = self._env
    sel = env.selection
    interface = env.mujoco_interface
    
    # Controller creation
    self._damping = Damping(env.robot_config, kv=10)
    ctrlr = self._controller = OSC(
      env.robot_config,
      kp=200,
      null_controllers=[self._damping],
      vmax=[0.5, 0.5],  # [m/s, rad/s]
      # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
      ctrlr_dof=[True, True, True, True, True, True],
      orientation_algorithm=1,
    )

    self._executor = Executor(interface, -0.12)

    if sel[1] == 'mug':
      place_dz = MUG_PLACE_DZ
    elif sel[1] == 'plate':
      place_dz = PLATE_PLACE_DZ
    elif sel[1] == 'bowl':
      place_dz = BOWL_PLACE_DZ

    if sel[0] == 'mug':
      mug_scale1 = env.scales['mug_mesh']
      stack(self._executor, interface, ctrlr, target_name='mug', container_name=sel[1], pickup_dz=MUG_PICKUP_DZ, pickup_dx=MUG_PICKUP_DX * mug_scale1[1], place_dz=place_dz, place_dx=MUG_PICKUP_DX * mug_scale1[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=True)
    else:
      bowl_scale1 = env.scales['bowl_mesh']
      stack(self._executor, interface, ctrlr, target_name='bowl', container_name=sel[1], pickup_dz=BOWL_PICKUP_DZ / bowl_scale1[1], pickup_dx=BOWL_PICKUP_DX * bowl_scale1[1], place_dz=place_dz, place_dx=BOWL_PICKUP_DX * bowl_scale1[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=True)
  
    self._executor.reset()

  def _action(self, time_step: ts.TimeStep, policy_state: types.NestedArray):
    if time_step.is_first():
      self.reset()
    
    action = self._executor.execute()

    if action is not None:
      action = action.astype(np.float32)

    return policy_step.PolicyStep(action=action)
