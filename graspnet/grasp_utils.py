"""Dynamically generate grasp labels during training.
Author: chenxi-wang
"""

import torch
import numpy as np
from knn_pytorch import knn_pytorch

from graspnet.loss import GRASP_MAX_WIDTH
from graspnet.data_utils import transform_point_cloud


def knn(ref, query, k=1):
    """Compute k nearest neighbors for each query point."""
    device = ref.device
    ref = ref.float().to(device)
    query = query.float().to(device)
    inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
    knn_pytorch.knn(ref, query, inds)
    return inds


def generate_grasp_views(N=300, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3), r=1):
    """View sampling on a unit sphere using Fibonacci lattices.
    Ref: https://arxiv.org/abs/0912.4540

    Input:
        N: [int]
            number of sampled views
        phi: [float]
            constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
        center: [np.ndarray, (3,), np.float32]
            sphere center
        r: [float]
            sphere radius

    Output:
        views: [torch.FloatTensor, (N,3)]
            sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """Transform approach vectors and in-plane rotation angles to rotation matrices.

    Input:
        batch_towards: [torch.FloatTensor, (N,3)]
            approach vectors in batch
        batch_angle: [torch.floatTensor, (N,)]
            in-plane rotation angles in batch

    Output:
        batch_matrix: [torch.floatTensor, (N,3,3)]
            rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = torch.norm(axis_y, dim=-1) == 0
    axis_y[mask_y, 1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


def process_grasp_labels(end_points):
    """Process labels according to scene points and object poses."""
    clouds = end_points["input_xyz"]  # (B, N, 3)
    seed_xyzs = end_points["fp2_xyz"]  # (B, Ns, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_views = []
    batch_grasp_views_rot = []
    batch_grasp_labels = []
    batch_grasp_offsets = []
    batch_grasp_tolerance = []

    for i in range(len(clouds)):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        poses = end_points["object_poses_list"][i]  # [(3, 4),]

        # get merged grasp points for label computation
        grasp_points_merged = []
        grasp_views_merged = []
        grasp_views_rot_merged = []
        grasp_labels_merged = []
        grasp_offsets_merged = []
        grasp_tolerance_merged = []

        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points["grasp_points_list"][i][obj_idx]  # (Np, 3)
            grasp_labels = end_points["grasp_labels_list"][i][obj_idx]  # (Np, V, A, D)
            grasp_offsets = end_points["grasp_offsets_list"][i][obj_idx]  # (Np, V, A, D, 3)
            grasp_tolerance = end_points["grasp_tolerance_list"][i][obj_idx]  # (Np, V, A, D)
            _, V, A, D = grasp_labels.size()
            num_grasp_points = grasp_points.size(0)
            # generate and transform template grasp views
            grasp_views = generate_grasp_views(V).to(pose.device)  # (V, 3)
            # grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")
            # grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], "3x3")
            grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], "3x3")

            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

            # assign views
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
            grasp_views_trans = torch.index_select(grasp_views_trans, 0, view_inds)  # (V, 3)
            grasp_views_trans = grasp_views_trans.unsqueeze(0).expand(num_grasp_points, -1, -1)  # (Np, V, 3)
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # (V, 3, 3)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(
                num_grasp_points, -1, -1, -1
            )  # (Np, V, 3, 3)
            grasp_labels = torch.index_select(grasp_labels, 1, view_inds)  # (Np, V, A, D)
            grasp_offsets = torch.index_select(grasp_offsets, 1, view_inds)  # (Np, V, A, D, 3)
            grasp_tolerance = torch.index_select(grasp_tolerance, 1, view_inds)  # (Np, V, A, D)
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_views_merged.append(grasp_views_trans)
            grasp_views_rot_merged.append(grasp_views_rot_trans)
            grasp_labels_merged.append(grasp_labels)
            grasp_offsets_merged.append(grasp_offsets)
            grasp_tolerance_merged.append(grasp_tolerance)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        grasp_views_merged = torch.cat(grasp_views_merged, dim=0)  # (Np', V, 3)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # (Np', V, 3, 3)
        grasp_labels_merged = torch.cat(grasp_labels_merged, dim=0)  # (Np', V, A, D)
        grasp_offsets_merged = torch.cat(grasp_offsets_merged, dim=0)  # (Np', V, A, D, 3)
        grasp_tolerance_merged = torch.cat(grasp_tolerance_merged, dim=0)  # (Np', V, A, D)

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Ns)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Np')
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1  # (Ns)

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)  # (Ns, 3)
        grasp_views_merged = torch.index_select(grasp_views_merged, 0, nn_inds)  # (Ns, V, 3)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)  # (Ns, V, 3, 3)
        grasp_labels_merged = torch.index_select(grasp_labels_merged, 0, nn_inds)  # (Ns, V, A, D)
        grasp_offsets_merged = torch.index_select(grasp_offsets_merged, 0, nn_inds)  # (Ns, V, A, D, 3)
        grasp_tolerance_merged = torch.index_select(grasp_tolerance_merged, 0, nn_inds)  # (Ns, V, A, D)

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views.append(grasp_views_merged)
        batch_grasp_views_rot.append(grasp_views_rot_merged)
        batch_grasp_labels.append(grasp_labels_merged)
        batch_grasp_offsets.append(grasp_offsets_merged)
        batch_grasp_tolerance.append(grasp_tolerance_merged)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    batch_grasp_views = torch.stack(batch_grasp_views, 0)  # (B, Ns, V, 3)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)  # (B, Ns, V, 3, 3)
    batch_grasp_labels = torch.stack(batch_grasp_labels, 0)  # (B, Ns, V, A, D)
    batch_grasp_offsets = torch.stack(batch_grasp_offsets, 0)  # (B, Ns, V, A, D, 3)
    batch_grasp_tolerance = torch.stack(batch_grasp_tolerance, 0)  # (B, Ns, V, A, D)

    # process labels
    batch_grasp_widths = batch_grasp_offsets[:, :, :, :, :, 2]
    label_mask = (batch_grasp_labels > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)
    u_max = batch_grasp_labels.max()
    batch_grasp_labels[label_mask] = torch.log(u_max / batch_grasp_labels[label_mask])
    batch_grasp_labels[~label_mask] = 0
    batch_grasp_view_scores, _ = batch_grasp_labels.view(batch_size, num_samples, V, A * D).max(dim=-1)

    end_points["batch_grasp_point"] = batch_grasp_points
    end_points["batch_grasp_view"] = batch_grasp_views
    end_points["batch_grasp_view_rot"] = batch_grasp_views_rot
    end_points["batch_grasp_label"] = batch_grasp_labels
    end_points["batch_grasp_offset"] = batch_grasp_offsets
    end_points["batch_grasp_tolerance"] = batch_grasp_tolerance
    end_points["batch_grasp_view_label"] = batch_grasp_view_scores.float()

    return end_points


def match_grasp_view_and_label(end_points):
    """Slice grasp labels according to predicted views."""
    top_view_inds = end_points["grasp_top_view_inds"]  # (B, Ns)
    template_views_rot = end_points["batch_grasp_view_rot"]  # (B, Ns, V, 3, 3)
    grasp_labels = end_points["batch_grasp_label"]  # (B, Ns, V, A, D)
    grasp_offsets = end_points["batch_grasp_offset"]  # (B, Ns, V, A, D, 3)
    grasp_tolerance = end_points["batch_grasp_tolerance"]  # (B, Ns, V, A, D)

    B, Ns, V, A, D = grasp_labels.size()
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_view_grasp_labels = torch.gather(grasp_labels, 2, top_view_inds_).squeeze(2)
    top_view_grasp_tolerance = torch.gather(grasp_tolerance, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1, 1).expand(-1, -1, -1, A, D, 3)
    top_view_grasp_offsets = torch.gather(grasp_offsets, 2, top_view_inds_).squeeze(2)

    end_points["batch_grasp_view_rot"] = top_template_views_rot
    end_points["batch_grasp_label"] = top_view_grasp_labels
    end_points["batch_grasp_offset"] = top_view_grasp_offsets
    end_points["batch_grasp_tolerance"] = top_view_grasp_tolerance

    return top_template_views_rot, top_view_grasp_labels, top_view_grasp_offsets, top_view_grasp_tolerance, end_points
