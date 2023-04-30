from gym.envs import registration
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box, Text, Dict

from ibc.environments.stack_lite.stack_lite_metrics import AverageDistanceToPickup, AverageDistanceToGoal, AverageObjectDistanceToGoal, AverageSuccessMetric
from ibc.environments.utils.xml.parse_xml import parse_xml
from ibc.environments.utils.xml.tag_replacers import ColorTagReplacer, ScaleTagReplacer
from ibc.environments.utils.mujoco.mujoco_interface import Mujoco
from ibc.environments.utils.mujoco.my_mujoco_config import MujocoConfig
import ibc.environments.stack.stack_viz as stack_viz

import mujoco as mj

import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
import gin

obj_names = [
  'container',
  'target2',
  'bowl',
  'bowl_2',
  'bowl_3',
  'plate',
  'plate_2',
  'plate_3',
  'mug',
  'mug_2',
  'mug_3',
]

combos = [
  ['bowl', 'plate'],
  ['mug', 'plate'],
  ['mug', 'bowl'],
  ['bowl', 'mug'],
]

# Figure out how to filter out invalid physics
def random_place(model, qpos, objs):
  qpos = qpos.copy()
  random.shuffle(objs)

  def l2(pos1, pos2):
    assert len(pos1) == len(pos2)
    d = 0
    for i in range(len(pos1)):
      d = d + (pos1[i] - pos2[i]) ** 2
    d = d ** (1 / 2)
    return d

  point_list = []

  for obj in objs:
    tries = 0

    while tries < 100000:
      x = (np.random.rand(1) - 0.5) * 0.5 # [-0.25, 0.25]
      y = (np.random.rand(1) - 0.5) * 0.25 + 0.7475 - 0.4 # [0.2225, 0.4725]
      too_close = False
      for j in range(len(point_list)):
        if l2(point_list[j], (x, y)) < 0.2 or abs(point_list[j][0] - x) < 0.1:
          too_close = True
      if not too_close:
        point_list.append((x, y))
        qpos[model.joint(obj).qposadr] = x
        qpos[model.joint(obj).qposadr + 1] = y
        break
      tries += 1

    if tries >= 100000:
      qpos = random_place(model, qpos, objs)
      break
  
  return qpos

@gin.configurable
class StackLiteEnv(MujocoEnv):
  metadata = {
    "render_modes": [
      "human",
      "rgb_array", # only use this mode for now
      "depth_array",
      "single_rgb_array",
      "single_depth_array",
    ],
    "render_fps": 125,
  }

  def get_metrics(self, num_episodes):
    metrics = [
      AverageDistanceToPickup(self, buffer_size=num_episodes),
      AverageDistanceToGoal(self, buffer_size=num_episodes),
      AverageObjectDistanceToGoal(self, buffer_size=num_episodes),
      AverageSuccessMetric(self, buffer_size=num_episodes),
    ]
    success_metric = metrics[-1]
    return metrics, success_metric

  @gin.configurable
  def __init__(self, xml_path='ibc/environments/assets/my_models/ur5_robotiq85/ur5_tabletop_static.xml', max_episode_steps=800, goal_distance=0.08, random_place=True, **kwargs):
    if 'observation_space' not in kwargs:
      kwargs['observation_space'] = Dict({
        "rgb": Box(low=0, high=255, shape=(224,224,3), dtype=np.float32),
      })

    self.xml_path = xml_path
    self.goal_distance = goal_distance
    self.max_episode_steps = max_episode_steps
    self.random_place = random_place

    MujocoEnv.__init__(
      self,
      str(Path(os.getcwd()).joinpath(self.xml_path)),
      1,
      **kwargs
    )

    # Define action space
    self.action_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    # Interface creation
    self.robot_config = MujocoConfig(self.model, self.data)
    self.mujoco_interface = Mujoco(self.robot_config)
    self.mujoco_interface.connect(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'])

    # Set starting angles
    self.START_ANGLES = self.model.numeric("START_ANGLES").data
    self.init_qpos[self.mujoco_interface.joint_pos_addrs] = self.START_ANGLES

    # Misc
    self.done = False
    self.reset_counter = 0
    self.img_save_dir = None
    self.pos_log = []
  
  def step(self, a):
    self.do_simulation(a, self.frame_skip)
    self.steps += 1
    ob = self._get_obs()
    reward = self._get_reward()
    terminated = self.done or self.steps >= self.max_episode_steps
    return ob, reward, terminated, {}

  def reset_model(self):
    self.reset_counter += 1
    self.steps = 0

    self.selection = ['mug', 'plate']
    self.colors = {'bowl': ['cyan', '0.000 0.948 0.966 1'], 'bowl_2': ['red', '0.929 0.000 0.000 1'], 'bowl_3': ['yellow', '1.000 0.906 0.001 1'], 'plate': ['cyan', '0.037 1.000 1.000 1'], 'plate_2': ['yellow', '1.000 1.000 0.000 1'], 'plate_3': ['white', '0.957 0.965 0.970 1'], 'mug': ['red', '0.931 0.084 0.039 1'], 'mug_2': ['black', '0.074 0.000 0.031 1'], 'mug_3': ['blue', '0.084 0.065 0.908 1']}
    self.scales = {'bowl_mesh': ['large', 1.2466501768695342], 'bowl_2_mesh': ['medium', 1.0743564648843356], 'bowl_3_mesh': ['small', 0.8011659323748979], 'plate_mesh': ['medium', 0.9127324687494754], 'plate_2_mesh': ['small', 0.795823622955547], 'plate_3_mesh': ['small', 0.8057945152094104], 'mug_mesh': ['small', 0.76807460708674], 'mug_2_mesh': ['large', 1.151023174736166], 'mug_3_mesh': ['medium', 1.011275386058562]}

    # Reset metrics
    self.min_dist_to_pickup = np.inf
    self.min_dist_to_goal = np.inf
    self.obj_dist_to_goal = np.inf

    # Manual calls to reload model from XML and reset viewers
    self.close()
    self._initialize_simulation()
    self.renderer.reset()
    self.renderer.render_step()

    # Interface recreation
    self.robot_config = MujocoConfig(self.model, self.data)
    self.mujoco_interface = Mujoco(self.robot_config)
    self.mujoco_interface.connect(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'])

    # Randomize object positions
    if self.random_place:
      qpos = random_place(self.model, self.init_qpos, [
        'bowl',
        'plate',
        'mug'
      ])
    else:
      qpos = np.array([0, 0.4, -2.3, 0.41, 1.57, 0,
        0,  0,  0,  0,  0,  0,
        0, 10,  0.05, 1,  0,  0,
        0,  0, 11,  0.05, 1,  0,
        0,  0,  0.01323144, 0.40757647, 0.05, 1,
        0,  0,  0,  1, 10,  0.05,
        1,  0,  0,  0, -1, 10,
        0.05, 1,  0,  0,  0, -0.18131834,
        0.29774985, 0.05, 1,  0,  0,  0,
        -1, 11,  0.05, 1,  0,  0,
        0,  1, 12,  0.05, 1,  0,
        0,  0,  0.21672751, 0.39437535, 0.05, 1,
        0,  0,  0, -1,  9,  0.05,
        1,  0,  0,  0,  1,  9,
        0.05, 1,  0,  0,  0, ])
    self.set_state(qpos, self.init_qvel)
    self.done = False

    return self._get_obs()
  
  def indicate_terminated(self):
    self.done = True

  def _get_obs(self):
    self._get_viewer('rgb_array').render(224,224,camera_id=0)
    data = self._get_viewer('rgb_array').read_pixels(224, 224, depth=False)
    image = data[::-1, :, :]
    # self._get_viewer('human').render()
    # image = np.zeros((224, 224, 3))
    image = image.astype(np.float32) / 255.

    pos_obs = { 'ee': self.mujoco_interface.get_xyz('EE') }

    for sel in ['mug', 'bowl', 'plate']:
      pos_obs[sel] = self.mujoco_interface.get_xyz(sel)

    self.pos_log.append(pos_obs)

    return {
      "rgb": image,
    }
  
  def _get_reward(self):
    self.min_dist_to_pickup = min(
      self.min_dist_to_pickup,
      self.dist(self.mujoco_interface.get_xyz(self.selection[0]))
    )
    self.min_dist_to_goal = min(
      self.min_dist_to_goal,
      self.dist(self.mujoco_interface.get_xyz(self.selection[1]))
    )
    self.obj_dist_to_goal = np.linalg.norm(
      self.mujoco_interface.get_xyz(self.selection[0]) - self.mujoco_interface.get_xyz(self.selection[1]),
    )
    reward = 1 if self.obj_dist_to_goal < self.goal_distance else 0
    reward = 1
    return reward

  def dist(self, goal):
    dist = np.linalg.norm(goal - self.mujoco_interface.get_xyz('EE'))
    return dist
  
  @property
  def succeeded(self):
    return self.obj_dist_to_goal < self.goal_distance

  def viewer_setup(self):
    assert self.viewer is not None
    v = self.viewer
    v.cam.type = mj.mjtCamera.mjCAMERA_FIXED
    v.cam.fixedcamid = 0
  
  def set_img_save_dir(self, summary_dir):
    self.img_save_dir = os.path.join(summary_dir, 'imgs')
    os.makedirs(self.img_save_dir, exist_ok=True)

  def save_image(self, traj):
    if traj.is_last():
      assert self.img_save_dir is not None  # pytype: disable=attribute-error
      fig, _ = stack_viz.visualize(self.pos_log)
      filename = os.path.join(self.img_save_dir,  # pytype: disable=attribute-error
                              str(self.reset_counter).zfill(6)+'_2d.png')
      fig.savefig(filename)
      plt.close(fig)

registration.register(id='StackLite-v0', entry_point=StackLiteEnv)
