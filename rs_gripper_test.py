"""Gripper interaction demo.

This script illustrates the process of importing grippers into a scene and making it interact
with the objects with actuators. It also shows how to procedurally generate a scene with the
APIs of the MJCF utility functions.

Example:
    $ python run_gripper_test.py
"""

import time
import json
import random
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.grippers import InspireRightHand
from robosuite.models.objects import BoxObject

# from robosuite.renderers.viewer import OpenCVViewer, MjviewerRenderer  # git pull-ed robosuite
from robosuite.renderers.mjviewer.mjviewer_renderer import MjviewerRenderer  # pip-installed robosuite
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjcf_utils import new_actuator, new_joint

import robosuite.utils.transform_utils as T

WIDTH_ANGLE_JSON_PATH = "inspire_width_angle.json"


class InspireGripperControl:
    def __init__(self, width_angle_dict, actuator_ctrlrange):
        self.width_angle_dict = width_angle_dict

        # convert 6d to 12d: from robosuite/models/grippers/inspire_hands.py, InspireRightHand.format_action()
        self.map_6d_to_12d = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])

        # NOTE: hand models from different sources (e.g., robosuite, dexanygrasp, dex-retargeting)
        # have different control ranges -- Definitely one of noise sources.

        # For the robosuite, control range -- 0: fully open, max: fully closed
        # In [0, 1000] -- 0: fully closed, 1000: fully open
        self.control_range = (actuator_ctrlrange[:12, 0] - actuator_ctrlrange[:12, 1]) / 1000
        self.control_base = actuator_ctrlrange[:12, 1]

        self.grip_info = {}
        for grip_type in self.width_angle_dict:
            self.grip_info[grip_type] = {"valid_widths": []}

            for width in self.width_angle_dict[grip_type]:
                # if 6d is -1, then it's not valid
                action_6d = self.width_angle_dict[grip_type][width]["6d"]
                if action_6d[0] > -1:
                    self.grip_info[grip_type]["valid_widths"].append(width)
                    self.width_angle_dict[grip_type][width]["12d"] = self.convert_6d_to_12d(action_6d)

            self.grip_info[grip_type]["num_widths"] = len(self.grip_info[grip_type]["valid_widths"])
            self.grip_info[grip_type]["curr_idx"] = self.grip_info[grip_type]["num_widths"] // 2

    def convert_6d_to_12d(self, action_6d):
        return np.array(action_6d)[self.map_6d_to_12d] * self.control_range + self.control_base

    def get_action(self, grip_type, close_gripper):
        curr_idx = self.grip_info[grip_type]["curr_idx"]
        if close_gripper:
            curr_idx -= 1
            curr_idx = max(curr_idx, 0)
        else:
            curr_idx += 1
            curr_idx = min(curr_idx, self.grip_info[grip_type]["num_widths"] - 1)

        self.grip_info[grip_type]["curr_idx"] = curr_idx
        width_key = self.grip_info[grip_type]["valid_widths"][curr_idx]

        return self.width_angle_dict[grip_type][width_key]["12d"]


if __name__ == "__main__":
    with open(WIDTH_ANGLE_JSON_PATH, "r", encoding="utf-8") as f:
        width_angle_dict = json.load(f)

    ### For now, this script just works for the Tripod grip
    # grip_type = random.choice(list(width_angle_dict.keys()))
    grip_type = "Tripod"

    ### Getting the gripper quat
    # get the pos and rotation matrix of the gripper, from the min width
    min_width = list(width_angle_dict[grip_type].keys())[0]
    gripper_pos = width_angle_dict[grip_type][min_width]["translation"]
    gripper_rot = width_angle_dict[grip_type][min_width]["rotation"]

    switch_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    two_finger_rot = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]) @ switch_axis
    two_finger_trans = np.array([0, 0, 1.11])
    matrix_two_fingers = np.vstack(
        (np.hstack((two_finger_rot, np.array(two_finger_trans).reshape((3, 1)))), np.array((0, 0, 0, 1)))
    )
    matrix_inspire = np.vstack(
        (np.hstack((gripper_rot, np.array(gripper_pos).reshape((3, 1)))), np.array((0, 0, 0, 1)))
    )
    mat_two_finger_to_inspire = np.dot(matrix_two_fingers, np.linalg.inv(matrix_inspire))

    inspire_trans = mat_two_finger_to_inspire[:3, 3]
    inspire_rot = mat_two_finger_to_inspire[:3, :3]
    grip_quat = T.mat2quat(np.array(inspire_rot))  # xyzw

    ### Basic mujoco world building
    # start with an empty world
    world = MujocoWorldBase()

    # add a table
    arena = TableArena(table_full_size=(0.4, 0.4, 0.05), table_offset=(0, 0, 1.1), has_legs=False)
    world.merge(arena)

    # add an object for grasping
    mujoco_object = BoxObject(
        name="box", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001]
    ).get_obj()
    # Set the position of this object
    mujoco_object.set("pos", "0 0 1.11")
    # Add our object to the world body
    world.worldbody.append(mujoco_object)

    # add reference objects for x and y axes
    x_ref = BoxObject(
        name="x_ref", size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 1], obj_type="visual", joints=None
    ).get_obj()
    x_ref.set("pos", "0.2 0 1.105")
    world.worldbody.append(x_ref)
    y_ref = BoxObject(
        name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual", joints=None
    ).get_obj()
    y_ref.set("pos", "0 0.2 1.105")
    world.worldbody.append(y_ref)

    # add a gripper
    gripper = InspireRightHand()  # PandaGripper()  # InspireRightHand()  # RethinkGripper()
    gripper_cls_name = gripper.__class__.__name__.lower()

    # XML quat is [w, x, y, z], Mujoco quat is [x, y, z, w]
    # only base_quat is effective here. What's the use for eef_relative quat?

    ### Setting the gripper quat
    base_quat = T.convert_quat(
        np.fromstring(gripper.worldbody[0].attrib.get("quat", "1 0 0 0"), dtype=np.float64, sep=" "), to="xyzw"
    )  # [0.71, 0, 0, 0.71] = 90 deg around [1, 0, 0]

    # base_quat = np.array([0, 0, 0, 1])
    quat_inspire = T.quat_multiply(grip_quat, base_quat)

    # Adjust the angle a bit
    scaled_aa_y_n18 = -np.pi / 10 * np.array([0, 1, 0])
    quat_y_n18 = T.axisangle2quat(scaled_aa_y_n18)
    quat_inspire = T.quat_multiply(quat_y_n18, quat_inspire)

    quat_inspire = T.convert_quat(quat_inspire, to="wxyz").tolist()
    gripper.worldbody[0].attrib["quat"] = "{} {} {} {}".format(*quat_inspire)

    """
    # Grab gripper offset (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
    # This is the comopunded rotation with the base body and the eef body as well!
    base_quat = T.convert_quat(
        np.fromstring(gripper.worldbody[0].attrib.get("quat", "1 0 0 0"), dtype=np.float64, sep=" "),
        to="xyzw"
    )  # [0.71, 0, 0, 0.71] = 90 deg around [1, 0, 0]
    eef_element = find_elements(
        root=gripper.root, tags="body", attribs={"name": gripper.correct_naming("eef")}, return_first=True
    )
    eef_relative_quat = T.convert_quat(string_to_array(eef_element.get("quat", "1 0 0 0")), to="xyzw")
    # [0, -0.71, 0, 0.71] = -90 deg around [0, 1, 0]

    # gripper.rotation_offset = T.quat_multiply(eef_relative_quat, base_quat)

    line_site = find_elements(
        root=gripper.root, tags="site", attribs={"name": gripper.correct_naming("grip_site_cylinder")}, return_first=True
    )
    # line site quat is [-0.5 -0.5 -0.5 0.5], (wxyz)
    # the rotation offset is [0.5 -0.5 0.5 -0.5]
    """

    ### Creating the gripper body, which moves the gripper up and down
    # Create another body with a slider joint to which we'll add this gripper
    gripper_body = ET.Element("body", name="gripper_base")
    gripper_body.set("pos", "-0.05 0 1.3")
    if "panda" in gripper_cls_name or "rethink" in gripper_cls_name:
        gripper_body.set("quat", "0 0 1 0")  # flip z
    gripper_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 1", damping="50"))

    # Add the dummy body with the joint to the global worldbody
    world.worldbody.append(gripper_body)
    # Merge the actual gripper as a child of the dummy body
    world.merge(gripper, merge_body="gripper_base")
    # Create a new actuator to control our slider joint
    world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

    ### Simulation & rendering
    # start simulation
    model = world.get_model(mode="mujoco")
    gripper_control = InspireGripperControl(width_angle_dict, model.actuator_ctrlrange)

    sim = MjSim(model)
    env = SimpleNamespace(sim=sim)
    viewer = MjviewerRenderer(env)
    viewer.update()

    sim_state = sim.get_state()

    # for gravity correction
    gravity_corrected = ["gripper_z_joint"]
    _ref_joint_vel_indexes = [sim.model.get_joint_qvel_addr(x) for x in gravity_corrected]

    # Set gripper parameters
    gripper_z_id = sim.model.actuator_name2id("gripper_z")
    gripper_z_low = -0.05  # 0.07
    gripper_z_high = 0.07  # -0.02
    gripper_z_is_low = False

    gripper_jaw_ids = [sim.model.actuator_name2id(x) for x in gripper.actuators]

    # Rethink gripper
    # if "rethink" in gripper_cls_name:
    #     gripper_open = [-0.0115, 0.0115]
    #     gripper_closed = [0.020833, -0.020833]

    # elif "panda" in gripper_cls_name:
    #     gripper_open = [1, -1]
    #     gripper_closed = [-1, 1]

    # elif "inspire" in gripper_cls_name:
    #     gripper_open = 1
    #     gripper_closed = -1

    gripper_is_closed = True

    # hardcode sequence for gripper looping trajectory
    seq = [(False, False), (True, False), (True, True), (False, True)]

    sim.set_state(sim_state)
    step = 0
    T = 500
    while True:
        if step % 100 == 0:
            print("step: {}".format(step))

            # Get contact information
            for contact in sim.data.contact[0 : sim.data.ncon]:
                geom_name1 = sim.model.geom_id2name(contact.geom1)
                geom_name2 = sim.model.geom_id2name(contact.geom2)
                if geom_name1 == "floor" and geom_name2 == "floor":
                    continue

                print("geom1: {}, geom2: {}".format(geom_name1, geom_name2))
                print("contact id {}".format(id(contact)))
                print("friction: {}".format(contact.friction))
                print("normal: {}".format(contact.frame[0:3]))

        # Iterate through gripping trajectory
        if step % T == 0:
            plan = seq[int(step / T) % len(seq)]
            gripper_z_is_low, gripper_is_closed = plan
            print("changing plan: gripper low: {}, gripper closed {}".format(gripper_z_is_low, gripper_is_closed))

        # Control gripper
        if gripper_z_is_low:
            sim.data.ctrl[gripper_z_id] = gripper_z_low
        else:
            sim.data.ctrl[gripper_z_id] = gripper_z_high

        if "inspire" in gripper_cls_name:
            gripper_action = gripper_control.get_action("Distal_Type", gripper_is_closed)
            sim.data.ctrl[gripper_jaw_ids] = gripper_action
        else:
            raise NotImplementedError

        # Step through sim
        sim.step()
        sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[_ref_joint_vel_indexes]

        # viewer.render()
        viewer.update()
        time.sleep(0.01)

        step += 1
