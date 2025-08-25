# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# dim = 20

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


# def _diff_to_pose_single(pose, diff):
#     """
#     由当前位姿 pose 和单帧的增量 diff 计算下一帧位姿

#     pose: np.array shape (7,) => [x, y, z, roll, pitch, yaw, gripper]
#     diff: np.array shape (7,) => [dx, dy, dz, droll, dpitch, dyaw, dgripper]
    
#     return: np.array shape (7,) => 下一帧的 pose
#     """
#     # 平移部分直接相加
#     next_xyz = pose[:3] + diff[:3]

#     # 姿态部分通过欧拉角 -> 矩阵旋转组合
#     rot_current = Rotation.from_euler("xyz", pose[3:6], degrees=False)
#     rot_delta = Rotation.from_euler("xyz", diff[3:6], degrees=False)
#     rot_next = rot_delta @ rot_current  # 注意顺序：先转当前，再转增量
#     next_angles = rot_next.as_euler("xyz", degrees=False)

#     # 夹爪闭合度直接相加
#     next_grip = pose[6] + diff[6]

#     return np.concatenate([next_xyz, next_angles, [next_grip]], axis=0)

# def _diffs_to_poses(poses,diffs):
#     if diffs.shape[-1] == 14:
#         # dual arm
#         left_poses = _diff_to_pose_single(poses[:, :7],diffs[:, :7])
#         right_poses = _diff_to_pose_single(poses[:, 7:], diffs[:, 7:])
#         output_poses = np.concatenate([left_poses, right_poses], axis=-1)
#     else:
#         output_poses = _diff_to_pose_single(poses)
#     return output_poses


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean_pose=0.14,
    momentum_std_pose=0.014,
    momentum_mean_rot=0.2,
    momentum_std_rot=0.05,
    momentum_mean_gripper=0.0002,
    momentum_std_gripper=0.0001,
    samples=100,
    topk=10,
    max_pose=8,
    max_rot=1,
    max_gripper=0.003,
    axis={},
    objective=l1,
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 20]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.zeros((rollout, 20), device=context_frame.device)

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * max_pose,
            torch.ones((rollout, 6), device=context_frame.device) * max_rot,
            torch.ones((rollout, 1), device=context_frame.device) * max_gripper,
            torch.ones((rollout, 3), device=context_frame.device) * max_pose,
            torch.ones((rollout, 6), device=context_frame.device) * max_rot,
            torch.ones((rollout, 1), device=context_frame.device) * max_gripper
        ],
        dim=-1,
    )

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):

            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-max_pose, max=max_pose)
            action_samples[:, 3:9] = torch.clip(action_samples[:, 3:9], min=-max_rot, max=max_rot)
            action_samples[:, 9:10] = torch.clip(action_samples[:, 9:10], min=-max_gripper, max=max_gripper)
            action_samples[:, 10:13] = torch.clip(action_samples[:, 10:13], min=-max_pose, max=max_pose)
            action_samples[:, 13:19] = torch.clip(action_samples[:, 13:19], min=-max_rot, max=max_rot)
            action_samples[:, 19:] = torch.clip(action_samples[:, 19:], min=-max_gripper, max=max_gripper)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]

            action_samples = action_samples.unsqueeze(1)
            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj = sample_action_traj()
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean_pose) + mean[..., :3] * momentum_mean_pose,
                mean_selected_actions[..., 3:9] * (1.0 - momentum_mean_rot) + mean[..., 3:9] * momentum_mean_rot,
                mean_selected_actions[..., 9:10] * (1.0 - momentum_mean_gripper)+ mean[..., 9:10] * momentum_mean_gripper,
                mean_selected_actions[..., 10:13] * (1.0 - momentum_mean_pose) + mean[..., 10:13] * momentum_mean_pose,
                mean_selected_actions[..., 13:19] * (1.0 - momentum_mean_rot) + mean[..., 13:19] * momentum_mean_rot,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper) + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std_pose) + std[..., :3] * momentum_std_pose,
                std_selected_actions[..., 3:9] * (1.0 - momentum_std_rot) + std[..., 3:9] * momentum_std_rot,
                std_selected_actions[..., 9:10] * (1.0 - momentum_std_gripper) + std[..., 9:10] * momentum_std_gripper,
                std_selected_actions[..., 10:13] * (1.0 - momentum_std_pose) + std[..., 10:13] * momentum_std_pose,
                std_selected_actions[..., 13:19] * (1.0 - momentum_std_rot) + std[..., 13:19] * momentum_std_rot,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        logger.info(f"[{step}] new mean: {mean.sum(dim=0)} std:{std.sum(dim=0)}")

    new_action = mean[None, :]

    return new_action


def compute_new_pose(pose, action, rotation_transformer):
    """
    :param pose: [B, T=1, 20]
    :param action: [B, T=1, 20]

    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz_left = pose[:, :3] + action[:, :3]
    new_xyz_right = pose[:, 10:13] + action[:, 10:13]

    # -- compute delta theta for left arm
    thetas_left_6d = pose[:, 3:9]
    thetas_left = rotation_transformer.inverse(thetas_left_6d)
    delta_thetas_left_6d = action[:, 3:9]
    delta_thetas_left = rotation_transformer.inverse(delta_thetas_left_6d)
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas_left]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas_left]
    new_angle_left = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    new_angle_left = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in new_angle_left]
    new_angle_left = np.stack([d for d in new_angle_left], axis=0)
    new_angle_left = rotation_transformer.forward(new_angle_left)


    # -- compute delta theta for right arm
    thetas_right_6d = pose[:, 13:19]
    thetas_right = rotation_transformer.inverse(thetas_right_6d)
    delta_thetas_right_6d = action[:, 13:19]
    delta_thetas_right = rotation_transformer.inverse(delta_thetas_right_6d) 
    matrices_right = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas_right]
    delta_matrices_right = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas_right]
    new_angle_right = [delta_matrices_right[t] @ matrices_right[t] for t in range(len(matrices_right))]
    new_angle_right = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in new_angle_right]
    new_angle_right = np.stack([d for d in new_angle_right], axis=0)
    new_angle_right = rotation_transformer.forward(new_angle_right)

    # -- compute delta gripper
    gripper_left = pose[:, 9:10] + action[:, 9:10]
    gripper_left = np.clip(gripper_left, 0, 1)
    gripper_right = pose[:, 19:] + action[:, 19:]
    gripper_right = np.clip(gripper_right, 0, 1)

    # -- new pose
    new_pose = np.concatenate([new_xyz_left, new_angle_left, gripper_left, new_xyz_right, new_angle_right, gripper_right], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]