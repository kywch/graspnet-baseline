import time

import numpy as np
from scipy.spatial.transform import Rotation
from graspnetAPI import Grasp

import robosuite
import robosuite.utils.camera_utils as CU
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.models.robots.robot_model import register_robot
from robosuite.models.grippers import register_gripper

# Helper functions for graspnet
from lift_test import ExtendedCameraInfo, GraspNetRunner

# These are necessary for using custom robots and grippers
from rs_grip_types import CustomInspireRightHand, CustomPanda, show_grab_site

register_robot(CustomPanda)
ROBOT_CLASS_MAPPING["CustomPanda"] = FixedBaseRobot
register_gripper(CustomInspireRightHand)

GRAB_SITE_OFFSET = np.array([-0.01, 0.02, 0])  # in the grip frame


# Camera choices: ["frontview", "birdview", "agentview", "robot0_robotview", "robot0_eye_in_hand"]
def make_env(camera_name, camera_height, camera_width):
    assert camera_name in ["frontview", "birdview", "agentview", "robot0_robotview", "robot0_eye_in_hand"], (
        "camera_name must be one of ['frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand']"
    )

    robot = "CustomPanda"

    controller_config = robosuite.load_part_controller_config(default_controller="OSC_POSE")
    controller_config["input_type"] = "absolute"
    controller_config["input_ref_frame"] = "world"
    controller_config["damping_ratio"] = 3  # make robot slower
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
        camera_names=[camera_name],
        camera_depths=[True],
        camera_heights=[int(camera_height)],
        camera_widths=[int(camera_width)],
        control_freq=20,
    )

    return env


def env_step(env, grip_type, pose, num_steps):
    gripper = env.robots[0].gripper["right"]
    gripper.set_grip_type(grip_type)

    ori_mat = Rotation.from_rotvec(pose[3:6]).as_matrix()
    new_pos, new_ori_aa = gripper.get_eef_pose_for_grab(pose[:3], ori_mat)
    pose[:3] = new_pos
    pose[3:6] = new_ori_aa

    for _ in range(num_steps):
        env.step(pose)
        time.sleep(0.03)

        # Visualize the grab site
        grab_pos, grab_ori_mat = gripper.get_grab_site_from_curr_eef(env)
        show_grab_site(env, grab_pos, grab_ori_mat)


if __name__ == "__main__":
    camera_name, camera_height, camera_width = "agentview", 720, 1280
    # camera_name, camera_height, camera_width = "robot0_eye_in_hand", 720, 1280

    # Manually set workspace for the lift env
    workspace_mask = np.zeros((camera_height, camera_width)).astype(bool)
    workspace_mask[200:520, 400:880] = True

    env = make_env(camera_name, camera_height, camera_width)
    grip_type_list = list(env.robots[0].gripper["right"].grip_info.keys())

    camera = ExtendedCameraInfo(env.sim, camera_name, camera_height, camera_width)
    graspnet_runner = GraspNetRunner(camera, "checkpoint-rs.tar")

    # ready, approach, grasp, lift
    # The below poses will be overwritten by the graspnet
    ready_pose = np.array([0, 0, 1.2, 0, 0, 0, 0.5])  # gripper half open
    approach_pose = np.array([0, 0, 1.2, 0, 0, 0, 0.5])  # gripper half open
    grab_pose = np.array([0, 0, 1.2, 0, 0, 0, -1])  # close gripper
    lift_pose = np.array([0, 0, 1.0, 0, 0, 0, -1])  # close gripper

    while True:
        for grip_type in grip_type_list:
            print("\nPlaying grip:", grip_type)

            obs_dict = env.reset()

            # NOTE: gripper instance seems to change with env.reset()
            gripper = env.robots[0].gripper["right"]
            gripper.set_grip_type(grip_type)
            gripper.set_grab_site_offset(GRAB_SITE_OFFSET)

            obj_pos = obs_dict["object-state"][:3]  # in world frame
            color_map = obs_dict["{}_image".format(camera_name)][::-1] / 255.0
            depth_map = CU.get_real_depth_map(
                sim=env.sim, depth_map=obs_dict["{}_depth".format(camera_name)][::-1]
            ).squeeze()

            # The pixel coordinates of the lift target
            obj_pixel = camera.get_pixel_coords(obj_pos)

            while True:
                gg = graspnet_runner.get_grasp(color_map, depth_map, workspace_mask, obj_pixel)
                if len(gg) > 0:
                    gg.nms()
                    gg.sort_by_score()

                    # Apply grasp score threshold
                    if gg[0].score > 0.7:
                        break

            best_grasp = Grasp(gg[0].grasp_array)
            # To compensate the z-underestimation, go a bit deeper
            grasp_pos = best_grasp.translation + 0.03 * best_grasp.rotation_matrix[:, 0]
            grasp_ori_aa = Rotation.from_matrix(best_grasp.rotation_matrix).as_rotvec()

            ready_pose[:3] = best_grasp.translation - 0.2 * best_grasp.rotation_matrix[:, 0]
            ready_pose[3:6] = grasp_ori_aa

            approach_pose[:3] = grasp_pos
            approach_pose[3:6] = grasp_ori_aa

            grab_pose[:3] = grasp_pos
            grab_pose[3:6] = grasp_ori_aa

            lift_pose[:6] = ready_pose[:6]

            # Just to init the viewer
            env.step(np.zeros(7))
            input("Press Enter to continue...")

            ### Execute the grasp
            env_step(env, grip_type, ready_pose, 50)
            env_step(env, grip_type, approach_pose, 50)
            env_step(env, grip_type, grab_pose, 60)
            env_step(env, grip_type, lift_pose, 50)

            input("Press Enter to continue...")
