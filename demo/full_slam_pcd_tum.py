import argparse
import bisect
import csv
import math
from os.path import join

import numpy as np
import torch

from grid_opt.configs import *
from grid_opt.datasets.sdf_3d_lidar import PosedSdf3DLidar
from grid_opt.slam.fuser import Fuser
from grid_opt.slam.system import System
from grid_opt.utils.utils import cond_mkdir
from grid_opt.utils.utils_sdf import save_mesh

import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--pcd_dir', required=True, help='Directory with <timestamp>.pcd or <timestamp>.ply files.')
parser.add_argument('--tum', required=True, help='Initial TUM pose file matched exactly to PCD filename stems.')
parser.add_argument('--tum_gt', default=None, help='Optional TUM pose file for supervision/evaluation. Defaults to --tum.')
parser.add_argument('--config', type=str, default='./configs/lidar/ncd_quad.yaml')
parser.add_argument('--default_config', type=str, default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/slam/pcd_tum')
parser.add_argument('--run_name', type=str, default='test')
parser.add_argument('--num_frames', type=int, default=None)
parser.add_argument('--keyframe_csv', default=None, help='GLIM keyframe_index_to_input_index.csv with a timestamp column.')
parser.add_argument('--max_keyframes', type=int, default=2000, help='Evenly downsample selected MISO frames to this cap.')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--submap_size', type=int, default=None)
parser.add_argument('--align', action='store_true', help='Run MISO latent-space global submap alignment after incremental SLAM.')
parser.add_argument('--no_mesh', action='store_true')
parser.add_argument('--enable_visualizer', action='store_true')


def rotation_matrix_to_quat_xyzw(R):
    trace = np.trace(R)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def quat_xyzw_to_rotation_matrix(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


def pose_matrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def read_tum(filename):
    stamps = []
    poses = []
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = line.split()
            if len(values) < 8:
                print(f'warning: skip malformed TUM line {line_number}')
                continue
            stamp = values[0]
            tx, ty, tz, qx, qy, qz, qw = [float(v) for v in values[1:8]]
            R = quat_xyzw_to_rotation_matrix(qx, qy, qz, qw)
            stamps.append(stamp)
            poses.append(pose_matrix(R, [tx, ty, tz]))
    return stamps, poses


def slerp_quat_xyzw(q0, q1, alpha):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    return (math.sin(theta_0 - theta) / sin_theta_0) * q0 + (math.sin(theta) / sin_theta_0) * q1


def average_poses(T_left, T_right, alpha):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (1.0 - alpha) * T_left[:3, 3] + alpha * T_right[:3, 3]
    q_left = rotation_matrix_to_quat_xyzw(T_left[:3, :3])
    q_right = rotation_matrix_to_quat_xyzw(T_right[:3, :3])
    q = slerp_quat_xyzw(q_left, q_right, alpha)
    T[:3, :3] = quat_xyzw_to_rotation_matrix(q[0], q[1], q[2], q[3])
    return T


def write_tum_poses(filename, stamps, poses):
    with open(filename, 'w') as file:
        for stamp, T in zip(stamps, poses):
            qx, qy, qz, qw = rotation_matrix_to_quat_xyzw(T[:3, :3])
            t = T[:3, 3]
            file.write(
                f'{stamp} '
                f'{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} '
                f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n'
            )


def write_tum(filename, stamps, grid_atlas):
    with open(filename, 'w') as file:
        for kf_id in range(grid_atlas.num_keyframes):
            R, t = grid_atlas.updated_kf_pose_in_world(kf_id)
            R_np = R.detach().cpu().numpy()
            t_np = t.detach().cpu().numpy().reshape(3)
            qx, qy, qz, qw = rotation_matrix_to_quat_xyzw(R_np)
            file.write(
                f'{stamps[kf_id]} '
                f'{t_np[0]:.6f} {t_np[1]:.6f} {t_np[2]:.6f} '
                f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n'
            )


def optimized_pose_map(stamps, grid_atlas):
    poses = {}
    for kf_id in range(grid_atlas.num_keyframes):
        R, t = grid_atlas.updated_kf_pose_in_world(kf_id)
        poses[stamps[kf_id]] = pose_matrix(
            R.detach().cpu().numpy(),
            t.detach().cpu().numpy().reshape(3)
        )
    return poses


def read_keyframe_csv(filename):
    stamps = []
    with open(filename, newline='') as file:
        sample = file.readline()
        file.seek(0)
        if 'timestamp' in sample:
            reader = csv.DictReader(file)
            for row in reader:
                stamps.append(row['timestamp'])
        else:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                # BLSS/KVORG keyframe_index_to_input_index.csv has no header:
                # keyframe_index,input_index,timestamp
                if len(row) < 3:
                    raise ValueError(f'{filename} has a malformed keyframe row: {row}')
                stamps.append(row[2])
    return stamps


def evenly_downsample(values, max_count):
    if max_count is None or max_count <= 0 or len(values) <= max_count:
        return list(values)
    if max_count == 1:
        return [values[0]]
    indices = np.linspace(0, len(values) - 1, max_count)
    indices = sorted(set(int(round(i)) for i in indices))
    return [values[i] for i in indices]


def select_miso_stamps(args):
    if args.keyframe_csv is not None:
        stamps = read_keyframe_csv(args.keyframe_csv)
    else:
        stamps, _ = read_tum(args.tum)
    if args.num_frames is not None:
        stamps = stamps[:args.num_frames]
    return evenly_downsample(stamps, args.max_keyframes)


def write_selected_stamps(filename, stamps):
    with open(filename, 'w') as file:
        file.write('miso_index,timestamp\n')
        for i, stamp in enumerate(stamps):
            file.write(f'{i},{stamp}\n')


def propagate_all_frame_poses(all_stamps, all_init_poses, selected_stamps, selected_opt_poses):
    selected = []
    stamp_to_index = {stamp: i for i, stamp in enumerate(all_stamps)}
    for stamp in selected_stamps:
        if stamp in selected_opt_poses and stamp in stamp_to_index:
            selected.append((stamp_to_index[stamp], stamp))
    if not selected:
        raise ValueError('No selected optimized poses overlap the full TUM trajectory.')

    selected.sort()
    selected_indices = [idx for idx, _ in selected]
    selected_by_index = {idx: stamp for idx, stamp in selected}
    propagated = []

    for i, T_init_i in enumerate(all_init_poses):
        if i in selected_by_index:
            propagated.append(selected_opt_poses[selected_by_index[i]])
            continue

        right_pos = bisect.bisect_right(selected_indices, i)
        left_pos = right_pos - 1

        if left_pos < 0:
            right_idx, right_stamp = selected[right_pos]
            T_right = selected_opt_poses[right_stamp] @ np.linalg.inv(all_init_poses[right_idx]) @ T_init_i
            propagated.append(T_right)
        elif right_pos >= len(selected):
            left_idx, left_stamp = selected[left_pos]
            T_left = selected_opt_poses[left_stamp] @ np.linalg.inv(all_init_poses[left_idx]) @ T_init_i
            propagated.append(T_left)
        else:
            left_idx, left_stamp = selected[left_pos]
            right_idx, right_stamp = selected[right_pos]
            alpha = (i - left_idx) / max(1, right_idx - left_idx)
            T_left = selected_opt_poses[left_stamp] @ np.linalg.inv(all_init_poses[left_idx]) @ T_init_i
            T_right = selected_opt_poses[right_stamp] @ np.linalg.inv(all_init_poses[right_idx]) @ T_init_i
            propagated.append(average_poses(T_left, T_right, alpha))

    return propagated


def dataset_option(cfg, profile, key, default):
    dataset_cfg = cfg.get('dataset', {})
    profile_cfg = dataset_cfg.get(profile, {})
    return profile_cfg.get(key, dataset_cfg.get(key, default))


def create_dataset(cfg, selected_stamps, profile):
    return PosedSdf3DLidar(
        lidar_folder=cfg['dataset']['path'],
        pose_file_gt=cfg['dataset']['pose_gt'],
        pose_file_init=cfg['dataset']['pose_init'],
        num_frames=cfg['dataset']['num_frames'],
        trunc_dist=cfg['dataset']['trunc_dist'],
        frame_samples=dataset_option(cfg, profile, 'frame_samples', 2**20 if profile == 'tracking' else 2**12),
        frame_batchsize=dataset_option(cfg, profile, 'frame_batchsize', 2**14 if profile == 'tracking' else 2**10),
        voxel_size=dataset_option(cfg, profile, 'voxel_size', 0.6 if profile == 'tracking' else 0.08),
        near_surface_std=dataset_option(cfg, profile, 'near_surface_std', 0.1 if profile == 'tracking' else 0.25),
        near_surface_n=dataset_option(cfg, profile, 'near_surface_n', 0 if profile == 'tracking' else 4),
        free_space_n=dataset_option(cfg, profile, 'free_space_n', 0 if profile == 'tracking' else 2),
        behind_surface_n=dataset_option(cfg, profile, 'behind_surface_n', 0 if profile == 'tracking' else 1),
        min_dist_ratio=dataset_option(cfg, profile, 'min_dist_ratio', 0.50),
        min_z=dataset_option(cfg, profile, 'min_z', -10.0),
        max_z=dataset_option(cfg, profile, 'max_z', 60.0),
        min_range=dataset_option(cfg, profile, 'min_range', 1.5),
        max_range=dataset_option(cfg, profile, 'max_range', 60.0),
        adaptive_range=dataset_option(cfg, profile, 'adaptive_range', False),
        pose_format='tum',
        match_pcd_by_tum_stamp=True,
        selected_stamps=selected_stamps
    )


def configure(args):
    cfg = load_config(args.config, args.default_config)
    if args.device is not None:
        cfg['device'] = args.device

    cfg['dataset']['path'] = args.pcd_dir
    cfg['dataset']['pose_init'] = args.tum
    cfg['dataset']['pose_gt'] = args.tum_gt if args.tum_gt is not None else args.tum
    cfg['dataset']['num_frames'] = args.num_frames

    cfg['tracking']['verbose'] = False
    cfg['mapping']['learning_rate'] = 1e-3
    cfg['mapping']['sigmoid_scale'] = 0.001 * 60.0
    cfg['mapping']['verbose'] = False
    cfg['mapping']['max_replay_frames'] = 5
    cfg['mapping']['max_replay_freq'] = 10

    cfg['system']['log_dir'] = join(args.save_dir, args.run_name)
    if args.submap_size is not None:
        cfg['system']['submap_size'] = args.submap_size
    cfg['visualizer']['enable'] = args.enable_visualizer
    return cfg


def main():
    args = parser.parse_args()
    cfg = configure(args)
    log_dir = join(args.save_dir, args.run_name)
    cond_mkdir(log_dir)
    selected_stamps = select_miso_stamps(args)
    print(f'selected MISO keyframes: {len(selected_stamps)}')
    selected_csv = join(log_dir, 'selected_miso_keyframes.csv')
    write_selected_stamps(selected_csv, selected_stamps)

    dataset_track = create_dataset(
        cfg,
        selected_stamps,
        profile='tracking'
    )
    dataset_map = create_dataset(
        cfg,
        selected_stamps,
        profile='mapping'
    )

    cfg['model']['pose']['num_poses'] = dataset_map.num_kfs
    grid_atlas = GridAtlas(cfg['model'], device=cfg['device'], dtype=torch.float32)
    grid_atlas.to(cfg['device'])

    system = System(
        model=grid_atlas,
        dataset_track=dataset_track,
        dataset_map=dataset_map,
        cfg=cfg,
        verbose=True
    )
    system.run()

    if args.align:
        fuser = Fuser(model=grid_atlas, dataset=dataset_map, cfg=cfg)
        fuser.align()

    model_path = join(log_dir, 'odometry.pth')
    torch.save(grid_atlas, model_path)
    out_tum = join(log_dir, 'optimized.tum')
    write_tum(out_tum, dataset_map.pose_stamps_init, grid_atlas)
    all_stamps, all_init_poses = read_tum(args.tum)
    selected_opt_poses = optimized_pose_map(dataset_map.pose_stamps_init, grid_atlas)
    all_optimized = propagate_all_frame_poses(all_stamps, all_init_poses, dataset_map.pose_stamps_init, selected_opt_poses)
    out_all_tum = join(log_dir, 'optimized_all_frames.tum')
    write_tum_poses(out_all_tum, all_stamps, all_optimized)
    print('saved model:', model_path)
    print('saved selected keyframes:', selected_csv)
    print('saved optimized TUM:', out_tum)
    print('saved propagated all-frame TUM:', out_all_tum)

    if not args.no_mesh:
        mesh_path = join(log_dir, 'final_mesh.ply')
        save_mesh(grid_atlas, grid_atlas.global_bound(), mesh_path, resolution=512)
        print('saved mesh:', mesh_path)


if __name__ == '__main__':
    main()
