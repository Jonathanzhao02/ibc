import numpy as np

class GripperStatus:
  def __init__(self, gripper_status):
    self.gripper_status = gripper_status

  def get_gripper_status(self):
    return self.gripper_status

  def set_gripper_status(self, new_gripper_status):
    self.gripper_status = new_gripper_status


class Action:
  def __init__(self, interface, controller):
    self.interface = interface
    self.controller = controller
    return
  
  def execute(self):
    raise NotImplementedError


class MoveTo(Action):
  def __init__(self, interface, controller, target_func, gripper_control_func, time_limit=None, error_limit=None, on_finish=None):
    assert time_limit is not None or error_limit is not None, "at least 1 of time limit or error limit should be indicated"

    super().__init__(interface, controller)
    self.target_func = target_func
    self.gripper_control_func = gripper_control_func
    self._gripper = None
    self.time_limit = time_limit
    self.error_limit = error_limit
    self.on_finish = on_finish

  def _set_gripper(self, gripper):
    self._gripper = gripper
  
  def reset(self):
    self.time_step = 0
    self.target = None

  def execute(self):
    finished = False

    # Get Target
    self.target = self.target_func(self.interface)

    # Calculate Forces
    feedback = self.interface.get_feedback()
    u = self.controller.generate(
      q=feedback["q"],
      dq=feedback["dq"],
      target=self.target,
    )

    # Set gripper force
    if self._gripper is not None:
      u = self.gripper_control_func(u, self._gripper)

    # calculate time step
    self.time_step += 1
    # calculate error
    # ee_xyz = robot_config.Tx("EE", q=feedback["q"])
    ee_xyz = self.interface.get_xyz("EE")
    error = np.linalg.norm(ee_xyz - self.target[:3])

    # whether stop criterion has been reached
    if self.time_limit is not None:
      if self.time_step >= self.time_limit:
        finished = True
    if self.error_limit is not None:
      if error <= self.error_limit:
        finished = True
      
    if finished and self.on_finish is not None:
      self.on_finish()
    
    return u, finished
    

class Executor:
  def __init__(self, interface, start_gripper_status):
    self.interface = interface
    self.action_list = []
    self.start_gripper_status = start_gripper_status
    self.reset()
  
  def reset(self):
    self.finished = False

    for action in self.action_list:
      action.reset()
    
    self.actions = iter(self.action_list)
    self.gripper = GripperStatus(self.start_gripper_status)
    self.action = next(self.actions, None)
    self.next_action = next(self.actions, None)

  def append(self, action):
    if hasattr(action, '_set_gripper'):
      action._set_gripper(self.gripper)
    self.action_list.append(action)

  def execute(self):
    if not self.finished:
      if self.action is None:
        return None

      u, finished = self.action.execute()
      
      if finished:
        self.finished = True

      return u
    else:
      self.action = self.next_action
      self.next_action = next(self.actions, None)
      self.finished = False
      return self.execute()
