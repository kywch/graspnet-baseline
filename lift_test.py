import time

import torch
import numpy as np
from scipy.spatial.transform import Rotation

import open3d as o3d
from PIL import Image

import mujoco
from graspnetAPI import Grasp, GraspGroup

from graspnet.network import GraspNet, pred_decode
from graspnet.collision_detector import ModelFreeCollisionDetector
from graspnet.data_utils import CameraInfo, create_point_cloud_from_depth_image

import robosuite
import robosuite.utils.camera_utils as CU

from robosuite.utils.camera_utils import project_points_from_world_to_camera as project_world_to_pixel
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.controllers import load_part_controller_config


# the distance threshold for a grasp candidate to be considered a valid grasp for the object (or clicked point)
PIXEL_DIST_THRESH = 50


# Camera choices: ["frontview", "birdview", "agentview", "robot0_robotview", "robot0_eye_in_hand"]
def make_env(camera_name, camera_height, camera_width):
    assert camera_name in ["frontview", "birdview", "agentview", "robot0_robotview", "robot0_eye_in_hand"], (
        "camera_name must be one of ['frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand']"
    )
    # Add a line

    robot = "Panda"

    controller_config = load_part_controller_config(default_controller="OSC_POSE")
    controller_config["input_type"] = "absolute"
    controller_config["input_ref_frame"] = "world"
    controller_config["damping_ratio"] = 3  # make robot slower
    controller_config = refactor_composite_controller_config(controller_config, robot, ["right"])

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


class ExtendedCameraInfo(CameraInfo):
    def __init__(self, sim, camera_name, camera_height, camera_width):
        camera_height = float(camera_height)
        camera_width = float(camera_width)
        intrinsic = CU.get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width)
        super().__init__(
            camera_width, camera_height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], scale=1.0
        )

        # camera frame (for depth map) <-> world frame <-> camera pixel
        self.world_to_pixel_mat = CU.get_camera_transform_matrix(sim, camera_name, camera_height, camera_width)
        self.camera_to_world_mat = CU.get_camera_extrinsic_matrix(sim, camera_name)
        self.camera_to_pixel_mat = self.world_to_pixel_mat @ self.camera_to_world_mat

    def get_pixel_coords(self, world_coords):
        return project_world_to_pixel(
            points=world_coords,
            world_to_camera_transform=self.world_to_pixel_mat,
            camera_height=self.height,
            camera_width=self.width,
        ).astype(np.int64)


class GraspNetRunner:
    def __init__(
        self,
        camera,
        graspnet_path,
        num_view=300,
        num_focus_sample=5000,
        num_random_sample=15000,
        pixel_dist_thresh=50,
        collision_voxel_size=0.01,
        collision_thresh=0.01,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_graspnet(graspnet_path, num_view)

        self.camera = camera
        self.world_to_pixel_mat = camera.world_to_pixel_mat
        self.camera_to_world_mat = camera.camera_to_world_mat
        self.camera_to_pixel_mat = camera.camera_to_pixel_mat

        # Vars for get_grasp
        self.num_focus_sample = num_focus_sample
        self.num_random_sample = num_random_sample
        self.pixel_dist_thresh = pixel_dist_thresh
        self.collision_voxel_size = collision_voxel_size
        self.collision_thresh = collision_thresh

        xmap = np.arange(self.camera.width)
        ymap = np.arange(self.camera.height)
        self._xmap, self._ymap = np.meshgrid(xmap, ymap)

    def _load_graspnet(self, graspnet_path, num_view):
        # Init the model
        self.network = GraspNet(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.network.to(self.device)

        # Load checkpointcloud_masked
        checkpoint = torch.load(graspnet_path)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print("-> loaded checkpoint %s (epoch: %d)" % (graspnet_path, start_epoch))

        # set model to eval mode
        self.network.eval()

    def get_grasp(self, color_map, depth_map, workspace_mask, focus_pixel, visualize=False):
        # Prepare the point cloud
        cloud = create_point_cloud_from_depth_image(depth_map, self.camera, organized=True)
        cloud_masked = cloud[workspace_mask]
        color_masked = color_map[workspace_mask]

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        # Sampling points
        focus_area = (
            np.sqrt((self._xmap - focus_pixel[1]) ** 2 + (self._ymap - focus_pixel[0]) ** 2) < self.pixel_dist_thresh
        )
        focus_cand = np.argwhere(focus_area[workspace_mask]).squeeze()
        assert len(focus_cand) > self.num_focus_sample, "not enough points in the focus area"
        focus_idxs = np.random.choice(focus_cand, self.num_focus_sample, replace=False)

        random_cand = np.argwhere(~focus_area[workspace_mask]).squeeze()
        assert len(random_cand) > self.num_random_sample, "not enough points in the random area"
        random_idxs = np.random.choice(random_cand, self.num_random_sample, replace=False)

        idxs = np.concatenate([focus_idxs, random_idxs], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # Get grasps
        with torch.no_grad():
            end_points = self.network(
                {
                    "point_clouds": torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device),
                    "cloud_colors": color_sampled,  # not used in the network
                }
            )
            grasp_preds = pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)

        # Filter grasps with the point(or object) of interest
        # by back projecting the grasp centers (camera frame) to the pixels
        grasp_pixels = project_world_to_pixel(
            points=gg.translations,
            world_to_camera_transform=self.camera_to_pixel_mat,
            camera_height=self.camera.height,
            camera_width=self.camera.width,
        ).astype(np.int64)
        distance_mask = np.linalg.norm(grasp_pixels - focus_pixel, axis=1) < PIXEL_DIST_THRESH
        gg = gg[distance_mask]

        if visualize:
            self.visualize_grasp_points(depth_map, grasp_pixels, focus_pixel, distance_mask)

        # Collision detection
        mfcdetector = ModelFreeCollisionDetector(np.array(cloud_o3d.points), self.collision_voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]

        if len(gg) == 0:
            return gg

        # Transform grasps to the world frame, then filter based on approach vector
        gg.transform(self.camera_to_world_mat)

        # Filter grasps with approach vector
        approach_vectors = gg.rotation_matrices[:, :, 0]
        assert np.abs(np.linalg.norm(approach_vectors[0]) - 1) < 1e-3, "Approach vector must be unit vector"
        cos_angle = np.arccos(np.clip(np.dot(approach_vectors, np.array([0, 0, -1])), -1, 1))
        approach_mask = np.abs(np.degrees(cos_angle)) < 30
        gg = gg[approach_mask]

        return gg

    def visualize_grasp_points(self, depth_map, grasp_pixels, focus_pixel, distance_mask):
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        norm_depth = (norm_depth * 255).astype(np.uint8)[:, :, np.newaxis]
        norm_depth = np.repeat(norm_depth, 3, axis=2)

        # Color-marking the pixels
        norm_depth[focus_pixel[0], focus_pixel[1], 0] = 255  # red dot
        for gp, d in zip(grasp_pixels, distance_mask):
            if d:
                norm_depth[gp[0], gp[1], 1] = 255  # green dots
            else:
                norm_depth[gp[0], gp[1], :] = 255  # white dots

        pil_depth = Image.fromarray(norm_depth)
        pil_depth.show()


if __name__ == "__main__":
    camera_name, camera_height, camera_width = "agentview", 720.0, 1280.0
    # camera_name, camera_height, camera_width = "robot0_eye_in_hand", 720.0, 1280.0

    env = make_env(camera_name, camera_height, camera_width)
    obs_dict = env.reset()

    #############################################################################
    ### check transformations
    # camera frame (for depth map) <-> world frame <-> camera pixel
    camera = ExtendedCameraInfo(env.sim, camera_name, camera_height, camera_width)

    world_to_pixel = camera.world_to_pixel_mat
    pixel_to_world = np.linalg.inv(world_to_pixel)

    # (0, 0, 0) in world frame -> (719, 640) in pixel frame
    world_origin_in_pixel = project_world_to_pixel(
        points=np.zeros(3),  # world origin
        world_to_camera_transform=world_to_pixel,
        camera_height=camera_height,
        camera_width=camera_width,
    ).astype(np.int64)

    camera_to_world = camera.camera_to_world_mat

    # <camera mode="fixed" name="agentview" pos="0.5 0 1.35" quat="0.653 0.271 0.271 0.653"/>
    camera_origin_in_world = camera_to_world[:3, 3]

    # assert np.allclose(camera_origin_in_world, [0.5, 0, 1.35])  # only for agentview

    world_to_camera = np.linalg.inv(camera_to_world)
    # see https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    world_origin_in_camera = world_to_camera[:3, 3]  # ~ [0, 0.6, 1.31]

    # Use camera_to_world to transform camera depth map to world coordinates
    obj_pos = obs_dict["object-state"][:3]  # in world frame
    obj_pos_in_camera = world_to_camera.dot(np.array(obj_pos.tolist() + [1]))[:3]
    obj_pos_in_world = camera_to_world.dot(np.array(obj_pos_in_camera.tolist() + [1]))[:3]
    assert np.allclose(obj_pos_in_world, obj_pos)

    obj_pixel = obj_pixel_from_camera = project_world_to_pixel(
        points=obj_pos_in_camera,
        world_to_camera_transform=world_to_pixel @ camera_to_world,
        camera_height=camera_height,
        camera_width=camera_width,
    ).astype(np.int64)

    obj_pixel_from_world = camera.get_pixel_coords(obj_pos)

    assert np.allclose(obj_pixel_from_camera, obj_pixel_from_world)

    ### visual inspection
    depth_map = obs_dict["{}_depth".format(camera_name)][::-1]
    depth_map = CU.get_real_depth_map(sim=env.sim, depth_map=depth_map)  # in camera frame

    if False:
        norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        norm_depth = np.repeat((norm_depth * 255).astype(np.uint8), 3, axis=2)
        norm_depth[obj_pixel_from_camera[0], obj_pixel_from_camera[1], 0] = 255  # red dot

        pil_depth = Image.fromarray(norm_depth)
        pil_depth.show()

    estimated_obj_pos = CU.transform_from_pixels_to_world(
        pixels=obj_pixel_from_camera,
        depth_map=depth_map,
        camera_to_world_transform=pixel_to_world,
    )

    # the most we should be off by in the z-direction is 3^0.5 times the maximum half-size of the cube
    max_z_err = np.sqrt(3) * 0.022
    z_err = np.abs(obj_pos[2] - estimated_obj_pos[2])
    assert z_err < max_z_err

    print("obj pos: {}".format(obj_pos))
    print("estimated obj pos: {}".format(estimated_obj_pos))
    print("z err: {}".format(z_err))

    #############################################################################
    ### graspnet pipeline
    graspnet_runner = GraspNetRunner(camera, "checkpoint-rs.tar")

    color_map = obs_dict["{}_image".format(camera_name)][::-1] / 255.0
    depth_map = CU.get_real_depth_map(sim=env.sim, depth_map=obs_dict["{}_depth".format(camera_name)][::-1]).squeeze()

    # Manually set workspace for now
    workspace_mask = np.zeros_like(depth_map).astype(bool)
    workspace_mask[200:520, 400:880] = True

    ### Get grasps
    while True:
        gg = graspnet_runner.get_grasp(color_map, depth_map, workspace_mask, obj_pixel, visualize=True)
        if len(gg) > 0:
            gg.nms()
            gg.sort_by_score()

            # Apply grasp score threshold
            if gg[0].score > 0.7:
                break

    best_grasp = Grasp(gg[0].grasp_array)

    ### Motion planning -- target pos/ori
    # position x, y, z
    target_pos = best_grasp.translation.tolist()
    target_pos[2] -= 0.03  # go a bit deeper
    # target_pos[2] += 0.5

    # NOTE: OSC controller -- self.goal_ori = Rotation.from_rotvec(action[3:6]).as_matrix()
    # The action[3:6] must produce the same rotation matrix
    # It can be done with: Rotation.from_matrix(best_grasp.rotation_matrix).as_rotvec()

    switch_axis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # NOTE: when the rotation matrix is identity, the gripper should face toward the x-axis, flat on x-y plane
    # target_ori = Rotation.from_matrix(switch_axis).as_rotvec().tolist()
    target_ori = Rotation.from_matrix(best_grasp.rotation_matrix @ switch_axis).as_rotvec().tolist()
    # target_ori = [0, 0, 0]

    eef_target = np.array(target_pos + target_ori + [-1])

    env.step(np.zeros(7))  # to init the viewer
    viewer = env.viewer.viewer

    length = 0.2
    approach_start = obj_pos - length * best_grasp.rotation_matrix[:, 0]
    eef_target = np.array(approach_start.tolist() + target_ori + [-1])

    ### Visualize approach vector
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # mjGEOM_ARROW,
        size=[0.02, 0, 0],
        pos=obj_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 1, 0, 1]),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=0.01,
        from_=approach_start,
        to=obj_pos,
    )
    viewer.user_scn.ngeom = 1
    env.viewer.update()

    input("Press Enter to continue...")

    ### Execute the grasp
    # Step 1: Move the gripper to the approach vector
    for i in range(50):
        env.step(eef_target)
        time.sleep(0.05)

    # Step 2: Move the gripper to the target position
    eef_target = np.array(target_pos + target_ori + [-1])
    for i in range(60):
        env.step(eef_target)
        time.sleep(0.05)

    # Step 3: Close the gripper
    eef_target[-1] = 1
    for i in range(30):
        env.step(eef_target)
        time.sleep(0.05)

    # Step 4: Lift the object
    # zero_ori = Rotation.from_matrix(switch_axis).as_rotvec().tolist()
    # NOTE: zero_ori should make the gripper face toward the x-axis, flat on x-y plane,
    #       which is the default orientation in the AnyGrasp dataset
    # eef_target = np.array([0, 0, 1.2] + zero_ori + [1])
    eef_target = np.array([0, 0, 1.2, np.pi, 0, 0, 1])
    for i in range(50):
        env.step(eef_target)
        time.sleep(0.05)

    print()
