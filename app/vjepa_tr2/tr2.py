# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DistributedSampler, DataLoader, Dataset, Subset
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
import zarr.storage
from app.vjepa_tr2.utils import get_auto_index,load_hdf5
from filelock import FileLock
import zarr
import shutil
import multiprocessing
from tqdm import tqdm
import concurrent.futures
import cv2

from src.utils.replay_buffer import ReplayBuffer
from src.utils.transforms.rotation_transformer import RotationTransformer
from src.utils.imagecodecs_numcodecs import register_codecs, Jpeg2k
register_codecs()


_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    val_ratio=0.05,
):
    dataset = TR2VideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
    )
# 定义训练集和验证集的索引
    def train_val_split(dataset, val_ratio=0.2):
        num_samples = len(dataset)
        indices = list(range(num_samples))
        split = int(val_ratio * num_samples)

        if split == 0 and val_ratio > 0:
            split = 1  # 确保验证集至少有一个样本
        
        # 打乱索引
        np.random.shuffle(indices)
        
        # 分割索引
        train_indices, val_indices = indices[split:], indices[:split]
        
        return train_indices, val_indices

    # 分割数据集
    train_indices, val_indices = train_val_split(dataset,val_ratio=val_ratio)

    # 创建训练集和验证集的子集
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_subset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_subset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    val_data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=val_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    assert len(train_data_loader) > 0, "train_data_loader is empty"
    assert len(val_data_loader) > 0, "val_data_loader is empty"

    logger.info("VideoDataset unsupervised data loader created")

    return train_data_loader,val_data_loader,train_sampler, val_sampler, dataset

def _get_data_from_raw(raw_dir):

    max_idx = get_auto_index(raw_dir)
    assert max_idx > 0, f"No data found in {raw_dir}"
    data_list = []


    for i in tqdm(
        range(max_idx),
        desc="reading from raw data",
        leave=False,
        position=1,
    ):
        demo_data_dict = {}
        ep_path = f"{raw_dir}/episode_{i}.hdf5"
        f_data_dict, f_timestamps_dict, v, fps = load_hdf5(ep_path)
        assert v == "4.0", f"version {v} not supported,should be 4.0"
        action = np.concatenate(
            (f_data_dict["master_left"]["qpos"][:], f_data_dict["master_right"]["qpos"][:]), axis=1
        )

        robot_left_pos = np.array(f_data_dict["puppet_left"]["tcp_pose"])[:, :3]
        robot_left_rot_euler = np.array(f_data_dict["puppet_left"]["tcp_pose"])[:, 3:]/ 180 * np.pi
        obs_gripper_left = np.concatenate((action[:, 6:7][:1], action[:, 6:7][:-1]), axis=0)

        robot_right_pos = np.array(f_data_dict["puppet_right"]["tcp_pose"])[:, :3]
        robot_right_rot_euler = np.array(f_data_dict["puppet_right"]["tcp_pose"])[:, 3:]/ 180 * np.pi
        obs_gripper_right = np.concatenate((action[:, 13:14][:1], action[:, 13:14][:-1]), axis=0)
        high_image = raw_dir + f"/episode_{i}_cam_high.mp4"
        low_image = raw_dir + f"/episode_{i}_cam_low.mp4"
        robot_left_in_hand_image = raw_dir + f"/episode_{i}_cam_left_wrist.mp4"
        robot_right_in_hand_image = raw_dir + f"/episode_{i}_cam_right_wrist.mp4"


        action_tcp_left = np.concatenate((f_data_dict["puppet_left"]["tcp_pose"][1:],f_data_dict["puppet_left"]["tcp_pose"][-1:]), axis=0)
        action_tcp_right = np.concatenate((f_data_dict["puppet_right"]["tcp_pose"][1:],f_data_dict["puppet_right"]["tcp_pose"][-1:]), axis=0)
        action_tcp = np.concatenate((action_tcp_left[:],action[:, 6:7][:], action_tcp_right[:],action[:, 13:14]), axis=1)

        demo_data_dict["action"] = action_tcp
        demo_data_dict["action_joints"] = action

        demo_data_dict["robot_left_pos"] = robot_left_pos
        demo_data_dict["robot_left_rot_euler"] = robot_left_rot_euler
        demo_data_dict["robot_left_gripper"] = obs_gripper_left
        demo_data_dict["robot_right_pos"] = robot_right_pos
        demo_data_dict["robot_right_rot_euler"] = robot_right_rot_euler
        demo_data_dict["robot_right_gripper"] = obs_gripper_right
        demo_data_dict["cam_high_image"] = high_image
        demo_data_dict["cam_low_image"] = low_image
        demo_data_dict["cam_left_wrist_image"] = robot_left_in_hand_image
        demo_data_dict["cam_right_wrist_image"] = robot_right_in_hand_image
        data_list.append(demo_data_dict)
        del f_data_dict, f_timestamps_dict,robot_left_in_hand_image,robot_right_in_hand_image,high_image,low_image

    return data_list

def _poses_to_diffs_single(poses):
    xyz = poses[:, :3]  # shape [T, 3]
    thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

def _poses_to_diffs(poses):
    if poses.shape[-1] == 14:
        # dual arm
        left_diffs = _poses_to_diffs_single(poses[:, :7])
        right_diffs = _poses_to_diffs_single(poses[:, 7:])
        diffs = np.concatenate([left_diffs, right_diffs], axis=-1)
    else:
        diffs = _poses_to_diffs_single(poses)
    return diffs

def _convert_actions(raw_actions, abs_action, rotation_transformer):
    
    if abs_action == False:
        # relative action
        raw_actions = _poses_to_diffs(raw_actions)

    is_dual_arm = False
    if raw_actions.shape[-1] == 14:
        # dual arm
        raw_actions = raw_actions.reshape(-1,2,7)
        is_dual_arm = True

    pos = raw_actions[...,:3]
    rot = raw_actions[...,3:6]/ 180 * np.pi
    gripper = raw_actions[...,6:]
    rot_6d = rotation_transformer.forward(rot)
    b_this_data = rotation_transformer.inverse(rot_6d)
    delta = b_this_data - rot
    assert delta.sum() < 1e-6 ,"actionrot_euler transform failed"

    raw_actions = np.concatenate([
        pos, rot_6d, gripper
    ], axis=-1).astype(np.float32)
    
    if is_dual_arm:
        raw_actions = raw_actions.reshape(-1,20)
    actions = raw_actions
        
    return actions


def _convert_zcai_to_replay(store, camera_views, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, n_demo=100):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = camera_views
    lowdim_keys = list()
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    raw_dataset_path = os.path.dirname(dataset_path)+ "/raw"

    demos = _get_data_from_raw(raw_dataset_path)

    for key in demos[0].keys():
        if "action" not in key and "image" not in key:
            lowdim_keys.append(key)

    episode_ends = list()
    enpisode_lengths = list()
    prev_end = 0
    for i in range(n_demo):
        demo = demos[i]
        episode_length = demo['action'].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
        enpisode_lengths.append(episode_length)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)
    _ = meta_group.array('episode_lengths', enpisode_lengths,
            dtype=np.int64, compressor=None, overwrite=True)


    all_state = []
    all_action = []
    for idx in tqdm(range(len(demos)), desc="Loading lowdim data"):
        demo = demos[idx]
        # make state data"]
        this_data = rotation_transformer.forward(demo["robot_left_rot_euler"])
        b_this_data = rotation_transformer.inverse(this_data)
        delta = b_this_data - demo["robot_left_rot_euler"]
        assert delta.sum() < 1e-6 ,"robot_left_rot_euler transform failed"
        robot_left_rot_euler = this_data

        this_data = rotation_transformer.forward(demo["robot_right_rot_euler"])
        b_this_data = rotation_transformer.inverse(this_data)
        delta = b_this_data - demo["robot_right_rot_euler"]
        assert delta.sum() < 1e-6 ,"robot_right_rot_euler transform failed"
        robot_right_rot_euler = this_data

        state = np.concatenate([
            demo["robot_left_pos"],
            robot_left_rot_euler,
            demo["robot_left_gripper"],
            demo["robot_right_pos"],
            robot_right_rot_euler,
            demo["robot_right_gripper"],
        ], axis=1)
        raw_state = np.concatenate([
            demo["robot_left_pos"],
            demo["robot_left_rot_euler"],
            demo["robot_left_gripper"],
            demo["robot_right_pos"],
            demo["robot_right_rot_euler"],
            demo["robot_right_gripper"],
        ], axis=1)
        all_state.append(state.astype(np.float32))
        action = _convert_actions(
                            raw_actions=demo["action"],
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer
                        )
        if abs_action == True:
            all_action.append(action.astype(np.float32))
        else:
            state_diff_6d = _convert_actions(
                raw_actions=raw_state,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer
            )

            all_action.append(np.concatenate([state_diff_6d[:1],
                            action], axis=0).astype(np.float32))
        
    data_arr = data_group.create_dataset(
                name="observation.state",
                shape=(n_steps,all_state[0].shape[1]),
                chunks=(1,all_state[0].shape[1]),
                compressor=None,
                dtype=np.float32
            )
    this_data = np.concatenate(all_state, axis=0)
    assert this_data.shape[0]==n_steps, "state data length mismatch"
    data_arr[:] = np.concatenate(all_state, axis=0)
    action_zrr = data_group.create_dataset(
                name="action",
                shape=(n_steps,all_action[0].shape[1]),
                chunks=(1,all_action[0].shape[1]),
                compressor=None,
                dtype=np.float32
            )
    this_data = np.concatenate(all_action, axis=0)
    assert this_data.shape[0]==n_steps, "action data length mismatch"
    action_zrr[:] = this_data
    
    
    def img_copy(zarr_arr, zarr_idx, imgs_list, img_idx):
        try:
            zarr_arr[zarr_idx] = imgs_list[img_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False
    
    with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = 'obs/' + key
                c,target_h,target_w = (3,240,320)
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.create_dataset(
                    name=key,
                    shape=(n_steps,target_h,target_w,c),
                    chunks=(1,target_h,target_w,c),
                    compressor=this_compressor,
                    dtype=np.uint8
                )
                for episode_idx in range(n_demo):
                    demo = demos[episode_idx]
                    # hdf5_arr = demo['obs'][key]
                    imgs_list = _get_imgs_from_video(demo[key],target_size=(target_h, target_w))
                    assert demo['action'].shape[0] == len(imgs_list),f"length of cam {demo[key]} is not equal to lowdim length"
                    for img_idx in range(len(imgs_list)):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + img_idx
                        futures.add(
                            executor.submit(img_copy, 
                                img_arr, zarr_idx, imgs_list, img_idx))
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
 
def _get_imgs_from_video(video_path, target_size=(240, 320)):
    """
    读取视频文件并返回调整大小后的图像数组。

    参数:
        video_path (str): 视频文件的路径。
        target_size (tuple): 目标图像大小，格式为 (height, width)。

    返回:
        list: 包含每一帧图像的列表，每一帧图像是一个 NumPy 数组，形状为 (target_height, target_width, 3)，通道顺序为 BGR。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    frames = []

    # 逐帧读取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取到视频末尾，退出循环

        # 检查帧的形状
        if frame.shape[:2] != (480, 640):
            raise ValueError(f"帧的形状不是 (480, 640)，而是 {frame.shape[:2]}")

        # 调整帧的大小
        resized_frame = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        # 将调整大小后的帧添加到列表中
        frames.append(resized_frame)

    # 释放视频捕获对象
    cap.release()

    return frames


class TR2VideoDataset(Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["cam_high", "cam_low","cam_left_wrist", "cam_right_wrist"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        vfps = 4
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        self.clip_nframes= int(fpc * fstp)

        self.image_transform = transform
        self.camera_frame = camera_frame
        self.camera_views = camera_views
        self.episode_len = get_auto_index(data_path)

        replay_buffer = None
        abs_action = False
        self.rotation_transformer = RotationTransformer(
            from_rep='euler_angles', to_rep='rotation_6d', from_convention='XYZ') #, euler_order='XYZ') #tr2

        if abs_action == True:
            cache_zarr_path = data_path + f'.{self.episode_len}_abs_action.' + '.zarr.zip'
        else:
            cache_zarr_path = data_path + f'.{self.episode_len}.' + '.zarr.zip'
        cache_lock_path = cache_zarr_path + '.lock'
        print('Acquiring lock on cache.')
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                # cache does not exists
                try:
                    print('Cache does not exist. Creating!')
                    # store = zarr.DirectoryStore(cache_zarr_path)
                    replay_buffer = _convert_zcai_to_replay(
                        store=zarr.storage.MemoryStore(), 
                        camera_views=camera_views, 
                        dataset_path=data_path, 
                        abs_action=abs_action, 
                        rotation_transformer=self.rotation_transformer,
                        n_demo=self.episode_len)
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(
                            store=zip_store
                        )
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print('Loading cached ReplayBuffer from Disk.')
                with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore())
                print('Loaded!')
        self.replay_buffer = replay_buffer
        self.episode_ends = self.replay_buffer.meta["episode_ends"][:]
        self.episode_starts = np.insert(self.episode_ends[:-1], 0, 0)
        self.episode_lengths = self.replay_buffer.meta["episode_lengths"][:]

    def __getitem__(self, index):
        vlen = self.episode_lengths[index]
        if vlen < self.clip_nframes:
            raise ValueError(f"Video {index} is too short to extract a clip of length {self.clip_nframes}!")
        
        ef = np.random.randint(self.clip_nframes + self.episode_starts[index], self.episode_ends[index])
        sf = ef - self.clip_nframes
        indices = np.arange(sf, ef, dtype=np.int64)
        states = self.replay_buffer.data["observation.state"][indices][:: self.frameskip]
        states = np.concatenate([states[:,:3],states[:,9:10]],axis=1)
        actions = self.replay_buffer.data["action"][indices][:-1][:: self.frameskip]
        action_single=np.concatenate([actions[:,:3],actions[:,9:10]],axis=1)

        camera_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,))]  
        buffer = self.replay_buffer.data[camera_view][indices][:: self.frameskip]
        # 添加transform
        if self.image_transform is not None:
            buffer = self.image_transform(buffer)

        return buffer, action_single, states

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path):
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath)

        # -- randomly sample a camera view
        camera_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"]
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        ef = np.random.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
        # --
        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]
        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)
        actions = self.poses_to_diffs(states)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.image_transform is not None:
            buffer = self.image_transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return self.episode_len
