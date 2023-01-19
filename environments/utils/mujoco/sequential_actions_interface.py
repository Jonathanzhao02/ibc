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
    
    # abstract method in base class
    # https://stackoverflow.com/questions/4382945/abstract-methods-in-python
    def execute(self):
        raise NotImplementedError("Please Implemente this Method")


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

    def execute(self):
        time_step = 0
        while True:
            # Get Target
            target = self.target_func(self.interface)

            # Calculate Forces
            feedback = self.interface.get_feedback()
            u = self.controller.generate(
                q=feedback["q"],
                dq=feedback["dq"],
                target=target,
            )

            # Set gripper force
            if self._gripper is not None:
                u = self.gripper_control_func(u, self._gripper)

            # send forces into Mujoco, step the sim forward
            self.interface.send_forces(u, update_display=False)

            # calculate time step
            time_step += 1
            # calculate error
            # ee_xyz = robot_config.Tx("EE", q=feedback["q"])
            ee_xyz = self.interface.get_xyz("EE")
            error = np.linalg.norm(ee_xyz - target[:3])

            # whether stop criterion has been reached
            if self.time_limit is not None:
                if time_step >= self.time_limit:
                    break
            if self.error_limit is not None:
                if error <= self.error_limit:
                    break
            
        if self.on_finish is not None:
            self.on_finish()
        

class Executor:
    def __init__(self, interface, start_angles, start_gripper_status):
        self.interface = interface
        self.action_list = []
        interface.send_target_angles(start_angles)
        self.gripper = GripperStatus(start_gripper_status)

    def append(self, action):
        if hasattr(action, '_set_gripper'):
            action._set_gripper(self.gripper)
        self.action_list.append(action)

    def execute(self):
        for i in range(len(self.action_list)):
            self.current_action = self.action_list[i]
            self.action_list[i].execute()
