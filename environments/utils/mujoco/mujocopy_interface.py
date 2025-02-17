import mujoco_py as mjp
import numpy as np
import glfw
from mujoco_py.generated import const

from abr_control.utils import transformations

from abr_control.interfaces.interface import Interface


class MujocoPy(Interface):
    """An interface for MuJoCo using the mujoco-py package.

    Parameters
    ----------
    robot_config: class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    dt: float, optional (Default: 0.001)
        simulation time step in seconds
    visualize: boolean, optional (Default: True)
        turns visualization on or off
    create_offscreen_rendercontext: boolean, optional (Default: False)
        create the offscreen rendercontext behind the main visualizer
        (helpful for rendering images from other cameras without displaying them)
    """

    def __init__(
        self,
        robot_config,
        dt=0.001,
        visualize=True,
        create_offscreen_rendercontext=False,
        on_step=None,
    ):

        super().__init__(robot_config)

        self.dt = dt  # time step
        self.count = 0  # keep track of how many times send forces is called

        self.robot_config = robot_config
        # set the time step for simulation
        self.robot_config.model.opt.timestep = self.dt

        # turns the visualization on or off
        self.visualize = visualize
        # if we want the offscreen render context
        self.create_offscreen_rendercontext = create_offscreen_rendercontext

        # function to call on sim step
        # should take this interface as an argument
        self.on_step = on_step

        # records last forces sent to simulation
        self.u = None

    def connect(self, joint_names=None, camera_id=-1, **kwargs):
        """
        joint_names: list, optional (Default: None)
            list of joint names to send control signal to and get feedback from
            if None, the joints in the kinematic tree connecting the end-effector
            to the world are used
        camera_id: int, optional (Default: -1)
            the id of the camera to use for the visualization
        """
        self.sim = mjp.MjSim(self.robot_config.model)
        self.sim.forward()  # run forward to fill in sim.data
        model = self.sim.model
        self.model = model

        if joint_names is None:
            joint_ids, joint_names = self.get_joints_in_ee_kinematic_tree()
            # print(joint_ids, joint_names)
            exit()
        else:
            joint_ids = [model.joint_name2id(name) for name in joint_names]
        # print(joint_names, joint_ids)
        self.joint_pos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        self.joint_vel_addrs = [model.get_joint_qvel_addr(name) for name in joint_names]

        joint_pos_addrs = []
        for elem in self.joint_pos_addrs:
            if isinstance(elem, tuple):
                joint_pos_addrs += list(range(elem[0], elem[1]))
            else:
                joint_pos_addrs.append(elem)
        self.joint_pos_addrs = joint_pos_addrs

        joint_vel_addrs = []
        for elem in self.joint_vel_addrs:
            if isinstance(elem, tuple):
                joint_vel_addrs += list(range(elem[0], elem[1]))
            else:
                joint_vel_addrs.append(elem)
        self.joint_vel_addrs = joint_vel_addrs

        # Need to also get the joint rows of the Jacobian, inertia matrix, and
        # gravity vector. This is trickier because if there's a quaternion in
        # the joint (e.g. a free joint or a ball joint) then the joint position
        # address will be different than the joint Jacobian row. This is because
        # the quaternion joint will have a 4D position and a 3D derivative. So
        # we go through all the joints, and find out what type they are, then
        # calculate the Jacobian position based on their order and type.
        index = 0
        self.joint_dyn_addrs = []
        for ii, joint_type in enumerate(model.jnt_type):
            if ii in joint_ids:
                self.joint_dyn_addrs.append(index)
            if joint_type == 0:  # free joint
                # self.joint_dyn_addrs += [jj + index for jj in range(1, 6)]
                # index += 6  # derivative has 6 dimensions
                continue
            elif joint_type == 1:  # ball joint
                self.joint_dyn_addrs += [jj + index for jj in range(1, 3)]
                index += 3  # derivative has 3 dimension
            else:  # slide or hinge joint
                index += 1  # derivative has 1 dimensions

        # print('joint_pos_addrs', self.joint_pos_addrs)
        # print('joint_vel_addrs', self.joint_vel_addrs)
        # print('joint_dyn_addrs', self.joint_dyn_addrs)
        # print('index', index)


        # give the robot config access to the sim for wrapping the
        # forward kinematics / dynamics functions
        self.robot_config._connect(
            self.sim, self.joint_pos_addrs, self.joint_vel_addrs, self.joint_dyn_addrs
        )

        # if we want to use the offscreen render context create it before the
        # viewer so the corresponding window is behind the viewer
        if self.create_offscreen_rendercontext:
            self.offscreen = mjp.MjRenderContextOffscreen(self.sim, 0)

        # create the visualizer
        if self.visualize:
            self.viewer = mjp.MjViewer(self.sim, **kwargs)
            # if specified, set the camera
            if camera_id > -1:
                self.viewer.cam.type = const.CAMERA_FIXED
                self.viewer.cam.fixedcamid = camera_id
        else:
            self.viewer = None

        print("MuJoCo session created")

    def disconnect(self):
        """Stop and reset the simulation."""
        if self.viewer:
            glfw.destroy_window(self.viewer.window)
            del self.viewer
        self.sim.reset()
        del self.sim
        print("MuJoCO session closed...")

    def get_joints_in_ee_kinematic_tree(self):
        """Get the names and ids of joints connecting the end-effector to the world"""
        model = self.sim.model
        # get the kinematic tree for the arm
        joint_ids = []
        joint_names = []
        body_id = model.body_name2id("EE")
        # start with the end-effector (EE) and work back to the world body
        while model.body_parentid[body_id] != 0:
            jntadrs_start = model.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(model.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                tmp_names.append(model.joint_id2name(tmp_ids[-1]))
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = model.body_parentid[body_id]
        # flip the list so it starts with the base of the arm / first joint
        joint_names = joint_names[::-1]
        joint_ids = np.array(joint_ids[::-1])

        return joint_ids, joint_names

    def get_orientation(self, name, object_type="body"):
        """Returns the orientation of an object as the [w x y z] quaternion [radians]

        Parameters
        ----------
        name: string
            the name of the object of interest
        object_type: string, Optional (Default: body)
            The type of mujoco object to get the orientation of.
            Can be: mocap, body, geom, site
        """
        if object_type == "mocap":  # commonly queried to find target
            quat = self.sim.data.get_mocap_quat(name)
        elif object_type == "body":
            quat = self.sim.data.get_body_xquat(name)
        elif object_type == "geom":
            xmat = self.sim.data.get_geom_xmat(name)
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        elif object_type == "site":
            xmat = self.sim.model.get_site_xmat(name)
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        else:
            raise Exception(
                f"get_orientation for {object_type} object type not supported"
            )
        return np.copy(quat)

    def set_mocap_orientation(self, name, quat):
        """Sets the orientation of an object in the Mujoco environment

        Sets the orientation of an object using the provided Euler angles.
        Angles must be in a relative xyz frame.

        Parameters
        ----------
        name: string
            the name of the object of interest
        quat: np.array
            the [w x y z] quaternion [radians] for the object.
        """
        self.sim.data.set_mocap_quat(name, quat)

    def send_forces(self, u, update_display=True):
        """Apply the specified torque to the robot joints

        Apply the specified torque to the robot joints, move the simulation
        one time step forward, and update the position of the hand object.

        Parameters
        ----------
        u: np.array
            the torques to apply to the robot [Nm]
        update_display: boolean, Optional (Default:True)
            toggle for updating display
        """

        # NOTE: the qpos_addr's are unrelated to the order of the motors
        # NOTE: assuming that the robot arm motors are the first len(u) values
        # print('sim.data.ctrl shape vs u (from osc) shape', self.sim.data.ctrl.shape, u.shape)
        self.sim.data.ctrl[:] = u[:]
        self.u = u.copy()

        # move simulation ahead one time step
        self.sim.step()

        # # Update position of hand object
        # feedback = self.get_feedback()
        # hand_xyz = self.robot_config.Tx(name="EE", q=feedback["q"])
        # self.set_mocap_xyz("hand", hand_xyz)

        # # Update orientation of hand object
        # hand_quat = self.robot_config.quaternion(name="EE", q=feedback["q"])
        # self.set_mocap_orientation("hand", hand_quat)

        if self.visualize and update_display:
            self.viewer.render()
        
        if self.on_step:
            self.on_step(self)

        self.count += self.dt

    def set_external_force(self, name, u_ext):
        """
        Applies an external force to a specified body

        Parameters
        ----------
        u_ext: np.array([x, y, z, alpha, beta, gamma])
            external force to apply [Nm]
        name: string
            name of the body to apply the force to
        """
        self.sim.data.xfrc_applied[self.model.body_name2id(name)] = u_ext

    def send_target_angles(self, q):
        """Move the robot to the specified configuration.

        Parameters
        ----------
        q: np.array
            configuration to move to [radians]
        """

        self.sim.data.qpos[self.joint_pos_addrs] = np.copy(q)
        self.sim.forward()

    def set_joint_state(self, q, dq):
        """Move the robot to the specified configuration.

        Parameters
        ----------
        q: np.array
            configuration to move to [rad]
        dq: np.array
            joint velocities [rad/s]
        """

        self.sim.data.qpos[self.joint_pos_addrs] = np.copy(q)
        self.sim.data.qvel[self.joint_vel_addrs] = np.copy(dq)
        self.sim.forward()

    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """

        self.q = np.copy(self.sim.data.qpos[self.joint_pos_addrs])
        self.dq = np.copy(self.sim.data.qvel[self.joint_vel_addrs])

        return {"q": self.q, "dq": self.dq}

    def get_xyz(self, name, object_type="body"):
        """Returns the xyz position of the specified object

        name: string
            name of the object you want the xyz position of
        object_type: string
            type of object you want the xyz position of
            Can be: mocap, body, geom, site
        """
        if object_type == "mocap":  # commonly queried to find target
            xyz = self.sim.data.get_mocap_pos(name)
        elif object_type == "body":
            xyz = self.sim.data.get_body_xpos(name)
        elif object_type == "geom":
            xyz = self.sim.data.get_geom_xpos(name)
        elif object_type == "site":
            xyz = self.sim.data.get_site_xpos(name)
        else:
            raise Exception(f"get_xyz for {object_type} object type not supported")

        return np.copy(xyz)

    def set_mocap_xyz(self, name, xyz):
        """Set the position of a mocap object in the Mujoco environment.

        name: string
            the name of the object
        xyz: np.array
            the [x,y,z] location of the target [meters]
        """
        self.sim.data.set_mocap_pos(name, xyz)

