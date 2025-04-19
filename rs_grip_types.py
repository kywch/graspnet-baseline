import time
import json
import numpy as np

import mujoco

import robosuite
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.models.robots import Panda
from robosuite.models.robots.robot_model import register_robot
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.inspire_hands import InspireRightHand

import robosuite.utils.transform_utils as T
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Define custom robot
# Use keyboard, to control OSC delta controller  --- mink?
# Print out the 6 dof actions for the arm


class CustomInspireRightHand(InspireRightHand):
    def __init__(self, idn=0):
        super().__init__(idn)  # Use Rososuite's inspire right hand xml

        # Update visualization
        for site in ["ee_x", "ee_y", "ee_z", "grip_site_cylinder"]:
            site_id = self._sites.index(site)
            rgba = self._elements["sites"][site_id].attrib["rgba"][:-1]
            rgba += "1" if "ee" in site else "0"
            self._elements["sites"][site_id].attrib["rgba"] = rgba

        # NOTE: where to put this json file?
        with open("inspire_width_angle.json", "r", encoding="utf-8") as f:
            self._width_angle_dict = json.load(f)

        # Get control range
        ctrl_range = np.array(
            [[float(x) for x in ac.attrib["ctrlrange"].split(" ")] for ac in self._elements["actuators"]]
        )
        self.control_range = (ctrl_range[:, 0] - ctrl_range[:, 1]) / 1000
        self.control_base = ctrl_range[:, 1]

        # Convert 6d to 12d: from robosuite/models/grippers/inspire_hands.py, InspireRightHand.format_action()
        self._map_6d_to_12d = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])

        # Convert the json file finger pos (0, 1000) to the 12-dof control signal
        self.grip_info = {}
        for grip_type in self._width_angle_dict:
            self.grip_info[grip_type] = {"valid_widths": []}

            for width in self._width_angle_dict[grip_type]:
                # if 6d is -1, then it's not valid
                action_6d = self._width_angle_dict[grip_type][width]["6d"]
                if action_6d[0] > -1:
                    self.grip_info[grip_type]["valid_widths"].append(width)
                    self._width_angle_dict[grip_type][width]["12d"] = self._convert_6d_to_12d(action_6d)

            self.grip_info[grip_type]["idx_scale"] = (len(self.grip_info[grip_type]["valid_widths"]) - 1) / 2.0

        # Set the gripper type
        self._grip_type = "Tripod"
        assert self._grip_type in self.grip_info, "Gripper type {} not found in gripper info!".format(self._grip_type)

        self._grip_hmat = {}
        self._grip_hmat_inv = {}
        # See AnyDexGrasp mesh generation, open3d for this
        self.eef_to_wrist_hmat = np.eye(4)
        self.eef_to_wrist_hmat[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    def _convert_6d_to_12d(self, action_6d):
        # NOTE: It probably isn't linear like below. AnyDexGrasp used driver_routine_to_angle.xls to map
        return np.array(action_6d)[self._map_6d_to_12d] * self.control_range + self.control_base

    def set_grip_type(self, grip_type):
        assert grip_type in self.grip_info, "Gripper type {} not found in gripper info!".format(grip_type)
        self._grip_type = grip_type

    def get_grip_to_wrist_hmat(self, grip_type):
        action_to_idx = 10  # int(self.grip_info[self._grip_type]["idx_scale"])  # middle value
        width_key = self.grip_info[grip_type]["valid_widths"][action_to_idx]

        trans = np.array(self._width_angle_dict[grip_type][width_key]["translation"])
        trans[0] -= 0.0078  # subtract the ring of metal, so set the origin to the center of wrist

        # inverse rotation to make it grip -> wrist frame
        rot_mat = np.array(self._width_angle_dict[grip_type][width_key]["rotation"])
        # rot_mat = np.linalg.inv(rot_mat)

        return T.make_pose(trans, rot_mat)  # 4x4 mat

    def get_grip_hmat(self, grip_type):
        if grip_type not in self._grip_hmat:
            grip_hmat = self.eef_to_wrist_hmat @ self.get_grip_to_wrist_hmat(grip_type)
            self._grip_hmat[grip_type] = grip_hmat
        else:
            grip_hmat = self._grip_hmat[grip_type]
        return grip_hmat

    def get_grab_site_from_curr_eef(self, env, grip_type):
        ref_id = env.sim.model.site_name2id("gripper0_right_grip_site")
        curr_ee_hmat = T.make_pose(env.sim.data.site_xpos[ref_id], env.sim.data.site_xmat[ref_id].reshape((3, 3)))

        grip_hmat = self.get_grip_hmat(grip_type)

        grab_site_hmat = curr_ee_hmat @ grip_hmat
        grab_pos, grab_ori_quat = T.mat2pose(grab_site_hmat)
        grab_ori_mat = T.quat2mat(grab_ori_quat)

        return grab_pos, grab_ori_mat

    # TODO: complete this
    def get_eef_pose_for_grab(self, grab_pos, grab_ori_aa):
        """Given grab pos and ori_aa, return the eef pose to feed to the controller"""
        pass
        # grip_hmat = self.get_grip_hmat(self._grip_type)

        # target_quat = T.axisangle2quat(target_ori_aa)
        # target_hmat = T.make_pose(target_pos, T.quat2mat(target_quat))

        # # NOTE: These go to the controller, so that the grab site is at target_pos/ori
        # pos, ori_quat = T.mat2pose(target_hmat)
        # return pos, T.quat2axisangle(ori_quat)

    def format_action(self, action):
        assert len(action) == self.dof, "Action dimension {} does not match the gripper dof {}".format(
            len(action), self.dof
        )
        assert -1 <= action[0] <= 1, "Action value {} is not in [-1, 1]".format(action)

        action_to_idx = int((action[0] + 1) * self.grip_info[self._grip_type]["idx_scale"])
        width_key = self.grip_info[self._grip_type]["valid_widths"][action_to_idx]

        return self._width_angle_dict[self._grip_type][width_key]["12d"]

    @property
    def dof(self):
        return 1


# Same as the PandaDexRH
class CustomPanda(Panda):
    # pass
    @property
    def default_gripper(self):
        return {"right": "CustomInspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    """
    NOTE: gripper_mount_quat_offset is overwritten. The format is [w, x, y, z]
    self.gripper[arm].worldbody.find("body").attrib["quat"] = array_to_string(
                    custom_gripper_mount_quat_offset
                )    
    """

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}  # w, x, y, z


register_robot(CustomPanda)
ROBOT_CLASS_MAPPING["CustomPanda"] = FixedBaseRobot
register_gripper(CustomInspireRightHand)


def show_grab_site(env, grab_pos, grab_ori_mat):
    # mark the grab site
    viewer = env.viewer.viewer
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # mjGEOM_ARROW,
        size=[0.01, 0, 0],
        pos=grab_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 1, 0, 1]),
    )

    approach_vec = grab_ori_mat[:, 0]
    gripper_vec = grab_ori_mat[:, 1]

    # Approach vector
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        pos=grab_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([1, 0, 0, 1]),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=0.01,
        from_=grab_pos,
        to=grab_pos + 0.15 * approach_vec,
    )

    # Gripper closing
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[2],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        pos=grab_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 1, 0, 1]),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[2],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=0.01,
        from_=grab_pos,
        to=grab_pos + 0.05 * gripper_vec,
    )

    viewer.user_scn.ngeom = 3
    env.viewer.update()


# robot = "PandaDexRH"
robot = "CustomPanda"

controller_config = robosuite.load_part_controller_config(default_controller="OSC_POSE")
controller_config["input_type"] = "absolute"
controller_config["input_ref_frame"] = "world"
# controller_config["damping_ratio"] = 3  # make robot slower
controller_config = refactor_composite_controller_config(controller_config, robot, ["right"])
# Match to robosuite/controllers/config/robots/default_panda_dex.json
controller_config["body_parts"]["right"]["gripper"]["use_action_scaling"] = False

# NOTE: consider using mink or curobo along with the default OSC delta controller
env = robosuite.make(
    "Lift",
    robots=[robot],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_object_obs=True,
    use_camera_obs=True,
    control_freq=20,
)

env.reset()
env.step(np.zeros(7))

gripper = env.robots[0].gripper["right"]

# -1 is closed, 1 is open
gripper_seq = [-1.0] * 10 + list(np.arange(-1, 1, 0.042)) + [1.0] * 2 + list(np.arange(1, -1, -0.042))

# See env.robots[0].part_controllers for arm and hand control
ref_id = env.sim.model.site_name2id("gripper0_right_grip_site")
eef_pos = env.sim.data.site_xpos[ref_id]
eef_quat = T.mat2quat(env.sim.data.site_xmat[ref_id].reshape((3, 3)))
eef_ori_aa = T.quat2axisangle(eef_quat)

# Keep the gripper pos and ori constant
eef_pose = np.zeros(7)  # OSC_POSE
eef_pose[:3] = eef_pos
eef_pose[3:6] = eef_ori_aa

while True:
    for grip_type in list(gripper.grip_info.keys()):
        gripper.set_grip_type(grip_type)
        print("Playing grip:", grip_type)

        grab_pos, grab_ori_mat = gripper.get_grab_site_from_curr_eef(env, grip_type)
        show_grab_site(env, grab_pos, grab_ori_mat)

        for _ in range(3):
            for i in range(len(gripper_seq)):
                eef_pose[-1] = gripper_seq[i]
                env.step(eef_pose)
                time.sleep(0.02)

print("Done.")
