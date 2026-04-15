"""
metrabs_inference.py
--------------------
Extract 2D and 3D poses from multi-camera videos using MeTRAbs (bml_movi_87)
and save them in the 26-joint calibration format expected by the pipeline.

This script is designed to run in the 'metrabs' conda environment (Python 3.10,
TensorFlow) and is called from calibrate.sh via:
    conda run --no-banner -n metrabs_opensim python metrabs_inference.py ...

It replaces both rtmlib_inference.py (2D) and inference.py (3D lifting) in a
single step with better accuracy.

Usage:
    conda run --no-banner -n metrabs_opensim python metrabs_inference.py \
        --video_dir  ./my_videos \
        --calib_toml ./Calib_scene.toml \
        --output_dir ./data/A001_P001_G001 \
        --aid 1 --pid 1 --gid 1 \
        --batch_size 8
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import cv2
import imageio
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as tfhub
import cameralib

# ---------------------------------------------------------------------------
# 26-joint calibration skeleton (subset of bml_movi_87)
# 20 virtual joint centers + backneck + sternum + 4 foot markers
# This matches METRABS_KEY / METRABS_BML87_INDICES in core/skeletons.py
# ---------------------------------------------------------------------------
METRABS_BML87_INDICES = [67, 0, 70, 3, 69, 68, 76, 84, 72, 80, 77, 85, 74, 82, 73, 81, 75, 83, 71, 79, 78, 86, 21, 52, 23, 54]
N_CALIB_JOINTS = len(METRABS_BML87_INDICES)  # 26

# ---------------------------------------------------------------------------
# bml_movi_87 -> Halpe26 mapping (for scale_scene.py compatibility)
# Halpe26 indices: Head=17, Neck=18, MidHip=19, LBigToe=20, RBigToe=21,
#                  LSmallToe=22, RSmallToe=23, LHeel=24, RHeel=25
# ---------------------------------------------------------------------------
BML87_TO_HALPE26 = {
    0:  67,  # Nose      <- head
    1:  5,   # LEye      <- lfronthead
    2:  36,  # REye      <- rfronthead
    3:  6,   # LEar      <- lbackhead
    4:  37,  # REar      <- rbackhead
    5:  76,  # LShoulder <- lsho
    6:  84,  # RShoulder <- rsho
    7:  72,  # LElbow    <- lelb
    8:  80,  # RElbow    <- relb
    9:  77,  # LWrist    <- lwri
    10: 85,  # RWrist    <- rwri
    11: 73,  # LHip      <- lhip
    12: 81,  # RHip      <- rhip
    13: 75,  # LKnee     <- lkne
    14: 83,  # RKnee     <- rkne
    15: 71,  # LAnkle    <- lank
    16: 79,  # RAnkle    <- rank
    17: 67,  # Head      <- head
    18: 0,   # Neck      <- backneck
    19: 68,  # MidHip    <- mhip
    20: 23,  # LBigToe   <- ltoe
    21: 54,  # RBigToe   <- rtoe
    22: 22,  # LSmallToe <- lfifthmetatarsal
    23: 53,  # RSmallToe <- rfifthmetatarsal
    24: 21,  # LHeel     <- lhee
    25: 52,  # RHeel     <- rhee
}


def parse_toml(path):
    """Parse a TOML file."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)

    import re
    data = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^\[(.+)\]$', line)
            if m:
                current = m.group(1)
                data[current] = {}
                continue
            if current and '=' in line:
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()
                try:
                    data[current][key] = eval(val)
                except Exception:
                    data[current][key] = val.strip('"')
    return data


def get_intrinsics_from_toml(toml_path, cam_names):
    """Extract intrinsic matrices and distortion coefficients from TOML."""
    data = parse_toml(toml_path)
    K_list = []
    dist_list = []
    for cam_name in cam_names:
        if cam_name not in data:
            print(f"ERROR: Section '[{cam_name}]' not found in {toml_path}")
            sys.exit(1)
        sec = data[cam_name]
        K = np.array(sec["matrix"], dtype=np.float64)
        K_list.append(K)
        dist = np.array(sec.get("distortions", [0, 0, 0, 0, 0]), dtype=np.float64)
        dist_list.append(dist)
    return K_list, dist_list


def get_best_person(poses3d, poses2d, boxes, imshape=None):
    """Select the best person based on bounding box area, with quality filtering."""
    if len(boxes) == 0:
        return None, None, 0.0

    areas = boxes[:, 2] * boxes[:, 3]  # width * height
    best_idx = int(np.argmax(areas))
    conf = float(boxes[best_idx, 4])

    # Filter: bbox too small relative to image → likely false positive
    if imshape is not None:
        img_area = imshape[0] * imshape[1]
        if areas[best_idx] < 0.005 * img_area:  # < 0.5% of image
            return None, None, 0.0

    return poses3d[best_idx], poses2d[best_idx], conf


def undistort_points(pts_2d, K, dist_coeffs):
    """Undistort 2D points and reproject to pixel coords (pinhole).

    Args:
        pts_2d: (N, 2) array of 2D points in distorted pixel space
        K: (3, 3) intrinsic matrix
        dist_coeffs: distortion coefficients (k1, k2, p1, p2, ...)
    Returns:
        (N, 2) array of undistorted 2D points in pixel space
    """
    if dist_coeffs is None or not np.any(dist_coeffs):
        return pts_2d
    pts = pts_2d.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return undist.reshape(-1, 2).astype(np.float32)


def bml87_to_calib26(kp_bml87):
    """Extract the 26 calibration joints from bml_movi_87 (87 joints)."""
    return kp_bml87[METRABS_BML87_INDICES].astype(np.float32)


def bml87_to_halpe26(kp_bml87):
    """Convert bml_movi_87 keypoints to Halpe26 format."""
    n_dim = kp_bml87.shape[-1]
    kp_out = np.zeros((26, n_dim), dtype=np.float32)
    for h_idx, bml_idx in BML87_TO_HALPE26.items():
        kp_out[h_idx] = kp_bml87[bml_idx]
    return kp_out


def process_video(video_path, model, skeleton, camera, start_frame=None, end_frame=None, batch_size=8):
    """Run MeTRAbs on a video and return per-frame poses.

    Uses a streaming generator to avoid loading all frames into RAM.
    """
    reader = imageio.get_reader(video_path, 'ffmpeg')
    total_frames = reader.count_frames()
    imshape = reader.get_data(0).shape[:2]
    reader.close()

    sf = start_frame if start_frame is not None else 0
    ef = end_frame if end_frame is not None else total_frames - 1
    if sf > ef or sf < 0 or ef >= total_frames:
        print(f"ERROR: Invalid frame range ({sf} to {ef}) for {video_path}")
        return [], [], [], []

    n_frames = ef - sf + 1
    print(f"  Video: {os.path.basename(video_path)} ({imshape[1]}x{imshape[0]}, frames {sf}-{ef})")

    # Generator that yields frames one by one (no bulk RAM allocation)
    def frame_generator():
        rd = imageio.get_reader(video_path, 'ffmpeg')
        for idx, frame in enumerate(rd):
            if idx < sf:
                continue
            if idx > ef:
                break
            yield frame
        rd.close()

    frame_ds = tf.data.Dataset.from_generator(
        frame_generator,
        output_signature=tf.TensorSpec(shape=(imshape[0], imshape[1], 3), dtype=tf.uint8)
    ).batch(batch_size).prefetch(1)

    all_poses3d = []
    all_poses2d = []
    all_confidences = []
    frame_indices = list(range(sf, sf + n_frames))
    n_dark = 0

    n_batches = int(np.ceil(n_frames / batch_size))
    for frame_batch in tqdm(frame_ds, total=n_batches, desc="  MeTRAbs inference"):
        pred = model.detect_poses_batched(
            frame_batch,
            intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
            skeleton=skeleton,
        )

        for i, (boxes, poses3d, poses2d) in enumerate(
            zip(pred['boxes'], pred['poses3d'], pred['poses2d'])
        ):
            # --- Dark/black frame detection ---
            frame_brightness = float(tf.reduce_mean(
                tf.cast(frame_batch[i], tf.float32)
            ).numpy())
            if frame_brightness < 15:
                all_poses3d.append(None)
                all_poses2d.append(None)
                all_confidences.append(0.0)
                n_dark += 1
                continue

            boxes_np = boxes.numpy()
            p3d_best, p2d_best, conf = get_best_person(
                poses3d.numpy(), poses2d.numpy(), boxes_np, imshape=imshape
            )

            # --- Skeleton plausibility check ---
            if p2d_best is not None:
                spread = np.std(p2d_best, axis=0).mean()
                if spread < 20:  # all joints collapsed to same spot
                    p3d_best, p2d_best, conf = None, None, 0.0

            all_poses3d.append(p3d_best)
            all_poses2d.append(p2d_best)
            all_confidences.append(conf)

    if n_dark > 0:
        print(f"  Filtered {n_dark}/{n_frames} dark frames")

    return frame_indices, all_poses3d, all_poses2d, all_confidences


def smooth_keypoints(poses_list, window=11, polyorder=3):
    """Apply Savitzky-Golay temporal smoothing to a list of keypoint arrays.

    Args:
        poses_list: list of (N_joints, D) arrays (one per frame)
        window: filter window size (must be odd, >= polyorder+2)
        polyorder: polynomial order for the filter
    Returns:
        list of smoothed arrays (same shapes)
    """
    from scipy.signal import savgol_filter

    if len(poses_list) < window:
        return poses_list  # not enough frames to smooth

    arr = np.array(poses_list)  # (N_frames, N_joints, D)
    n_frames, n_joints, n_dim = arr.shape

    # Smooth each joint coordinate independently
    for j in range(n_joints):
        for d in range(n_dim):
            arr[:, j, d] = savgol_filter(arr[:, j, d], window, polyorder)

    return [arr[i] for i in range(n_frames)]


def save_json(filepath, frame_indices, poses, scores):
    """Save poses in the JSON format expected by the calibration pipeline."""
    data = []
    for fidx, pose, score in zip(frame_indices, poses, scores):
        data.append({
            "frame_index": int(fidx),
            "skeleton": [{
                "pose": pose.flatten().tolist(),
                "score": score.tolist(),
            }]
        })
    with open(filepath, "w") as f:
        json.dump({"data": data}, f, indent=2, ensure_ascii=True)


def save_skeleton_w(filepath, frame_indices, poses3d):
    """Save world skeleton JSON (using first camera's 3D as reference)."""
    skeleton = np.array(poses3d, dtype=np.float64)  # (N, N_joints, 3)
    out = {
        "frame_indices": [int(f) for f in frame_indices],
        "skeleton": skeleton.tolist(),
    }
    with open(filepath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2D+3D poses with MeTRAbs (bml_movi_87) for multi-camera calibration."
    )
    parser.add_argument("--video_dir", required=True, help="Folder containing camera video files")
    parser.add_argument("--calib_toml", required=True, help="TOML file with camera intrinsics")
    parser.add_argument("--output_dir", required=True, help="Output folder (e.g. ./data/A001_P001_G001)")
    parser.add_argument("--aid", type=int, default=1, help="Action ID")
    parser.add_argument("--pid", type=int, default=1, help="Person ID")
    parser.add_argument("--gid", type=int, default=1, help="Group/Scene ID")
    parser.add_argument("--subset_name", default="noise_1_0", help="Subset folder name")
    parser.add_argument("--start_frame", type=int, default=None, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--skeleton", default="bml_movi_87", help="MeTRAbs skeleton name")
    args = parser.parse_args()

    # Collect and sort video files
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.video_dir, ext)))
    video_files = sorted(video_files)

    if not video_files:
        print(f"ERROR: No video files found in {args.video_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s):")
    for i, v in enumerate(video_files):
        print(f"  [{i+1}] {os.path.basename(v)}")

    # Get camera names (same logic as calibrate.sh)
    cam_names = [os.path.splitext(os.path.basename(v))[0] for v in video_files]

    # Load intrinsics from TOML
    K_list, dist_list = get_intrinsics_from_toml(args.calib_toml, cam_names)
    print(f"\nLoaded intrinsics for {len(K_list)} cameras from {args.calib_toml}")

    # Load MeTRAbs model (this takes 30-60s: model loading + TF graph compilation)
    print(f"\nLoading MeTRAbs model (skeleton={args.skeleton}) — please wait...", flush=True)
    model = tfhub.load('https://bit.ly/metrabs_l')
    print("Model loaded. Running warmup inference...", flush=True)
    # Warmup: first call triggers TF graph compilation (slow), subsequent calls are fast
    _dummy = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    model.detect_poses_batched(tf.constant(_dummy), skeleton=args.skeleton)
    print("Warmup done. Starting pose extraction.\n", flush=True)

    # Create output directories
    out_2d_dir = os.path.join(args.output_dir, args.subset_name, "2d_joint")
    out_3d_dir = os.path.join(args.output_dir, args.subset_name, "3d_joint")
    out_halpe26_dir = os.path.join(args.output_dir, args.subset_name, "2d_joint_halpe26")
    os.makedirs(out_2d_dir, exist_ok=True)
    os.makedirs(out_3d_dir, exist_ok=True)
    os.makedirs(out_halpe26_dir, exist_ok=True)

    skeleton_w_data = None  # Will store first camera's 3D for skeleton_w

    for cam_idx, (video_path, K, dist) in enumerate(zip(video_files, K_list, dist_list), start=1):
        cid = cam_idx
        base_name = f"A{args.aid:03d}_P{args.pid:03d}_G{args.gid:03d}_C{cid:03d}.json"

        print(f"\n[Camera {cid}]")

        # Create cameralib Camera with real intrinsics
        camera = cameralib.Camera(
            intrinsic_matrix=K.astype(np.float32),
            distortion_coeffs=dist.astype(np.float32) if np.any(dist != 0) else None,
        )

        # Run inference
        frame_indices, poses3d_raw, poses2d_raw, confidences = process_video(
            video_path, model, args.skeleton, camera,
            start_frame=args.start_frame, end_frame=args.end_frame,
            batch_size=args.batch_size,
        )

        if not frame_indices:
            print(f"  WARNING: No frames processed for camera {cid}")
            continue

        # Temporal smoothing (Savitzky-Golay) to reduce frame-to-frame jitter
        # Only smooth frames where a person was detected
        valid_3d = [p for p in poses3d_raw if p is not None]
        valid_2d = [p for p in poses2d_raw if p is not None]
        if len(valid_3d) > 11:
            smoothed_3d = smooth_keypoints(valid_3d)
            smoothed_2d = smooth_keypoints(valid_2d)
            vi = 0
            for i in range(len(poses3d_raw)):
                if poses3d_raw[i] is not None:
                    poses3d_raw[i] = smoothed_3d[vi]
                    poses2d_raw[i] = smoothed_2d[vi]
                    vi += 1
            print(f"  Applied Savitzky-Golay smoothing ({len(valid_3d)} frames)")

        # Convert to full 87-joint format, Halpe26 for scaling compatibility
        full87_2d_list = []
        full87_3d_list = []
        halpe26_2d_list = []
        scores_2d_list = []
        scores_3d_list = []
        scores_halpe26_list = []

        N_FULL_JOINTS = 87

        has_distortion = np.any(dist != 0)
        if has_distortion:
            print(f"  Undistorting 2D keypoints (dist={dist[:4]}...)")

        # Get image dimensions for per-joint quality scoring
        _reader = imageio.get_reader(video_path, 'ffmpeg')
        img_h, img_w = _reader.get_data(0).shape[:2]
        _reader.close()

        for p3d, p2d, conf in zip(poses3d_raw, poses2d_raw, confidences):
            if p3d is not None:
                # Undistort raw 2D keypoints (all 87) before saving
                if has_distortion:
                    p2d = undistort_points(p2d, K, dist)

                # Keep all 87 joints directly (no subsetting)
                kp_2d = p2d.astype(np.float32)   # (87, 2)
                kp_3d = p3d.astype(np.float32)   # (87, 3)
                # Map to Halpe26 (for scale_scene.py backward compat)
                kp_2d_halpe26 = bml87_to_halpe26(p2d)

                # Per-joint 2D confidence scoring (all 87 joints)
                s2d = np.full(N_FULL_JOINTS, conf, dtype=np.float32)

                # Penalize 2D joints outside image bounds
                margin = 10
                oob = ((kp_2d[:, 0] < margin) | (kp_2d[:, 0] > img_w - margin) |
                       (kp_2d[:, 1] < margin) | (kp_2d[:, 1] > img_h - margin))
                s2d[oob] *= 0.1

                # 3D confidence: bbox conf only
                s3d = np.full(N_FULL_JOINTS, conf, dtype=np.float32)
                s_halpe26 = np.full(26, conf, dtype=np.float32)
            else:
                # No detection
                kp_2d = np.zeros((N_FULL_JOINTS, 2), dtype=np.float32)
                kp_3d = np.zeros((N_FULL_JOINTS, 3), dtype=np.float32)
                kp_2d_halpe26 = np.zeros((26, 2), dtype=np.float32)
                s2d = np.zeros(N_FULL_JOINTS, dtype=np.float32)
                s3d = np.zeros(N_FULL_JOINTS, dtype=np.float32)
                s_halpe26 = np.zeros(26, dtype=np.float32)

            full87_2d_list.append(kp_2d)
            full87_3d_list.append(kp_3d)
            halpe26_2d_list.append(kp_2d_halpe26)
            scores_2d_list.append(s2d)
            scores_3d_list.append(s3d)
            scores_halpe26_list.append(s_halpe26)

        # Save 2D (full 87-joint bml_movi_87)
        save_json(
            os.path.join(out_2d_dir, base_name),
            frame_indices, full87_2d_list, scores_2d_list
        )
        print(f"  Saved {len(frame_indices)} bml_movi_87 2D frames (87 joints) -> {out_2d_dir}/{base_name}")

        # Save 3D (full 87-joint bml_movi_87)
        save_json(
            os.path.join(out_3d_dir, base_name),
            frame_indices, full87_3d_list, scores_3d_list
        )
        print(f"  Saved {len(frame_indices)} bml_movi_87 3D frames (87 joints) -> {out_3d_dir}/{base_name}")

        # Save Halpe26 2D (for scale_scene.py)
        save_json(
            os.path.join(out_halpe26_dir, base_name),
            frame_indices, halpe26_2d_list, scores_halpe26_list
        )
        print(f"  Saved {len(frame_indices)} Halpe26 2D frames -> {out_halpe26_dir}/{base_name}")

        # Store first camera's 3D for skeleton_w
        if skeleton_w_data is None:
            skeleton_w_data = (frame_indices, full87_3d_list)

    # Save skeleton_w (reference 3D from first camera)
    if skeleton_w_data is not None:
        skel_path = os.path.join(args.output_dir, args.subset_name, f"skeleton_w_G{args.gid:03d}.json")
        save_skeleton_w(skel_path, skeleton_w_data[0], skeleton_w_data[1])
        print(f"\nSaved skeleton_w -> {skel_path}")

    print("\n" + "=" * 60)
    print("MeTRAbs pose extraction complete!")
    print(f"Output: {args.output_dir}/{args.subset_name}/")
    print(f"  2d_joint/        : bml_movi_87 2D poses (87 joints)")
    print(f"  3d_joint/        : bml_movi_87 3D poses (87 joints)")
    print(f"  2d_joint_halpe26/: Halpe26 2D poses (for scaling)")
    print("=" * 60)


if __name__ == "__main__":
    main()
