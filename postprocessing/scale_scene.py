"""
scale_scene.py
--------------
Scales and reorients a calibrated scene to a metric, gravity-aligned coordinate system.

This script performs two main operations:
1.  **Reorientation**: It defines a new coordinate system by fitting a plane to
    the person's feet keypoints to define the ground (XZ plane) and its normal
    as the vertical axis (Y-axis).
2.  **Scaling**: It scales the entire scene to metric units (meters) based on the
    person's real-world height.

The final transformed calibration is saved to a new JSON file and can be exported
to a TOML file.

Usage:
    python scale_scene.py \\
        --prefix ./data/A001_P003_G001 \\
        --calib linear_1_0_ba \\
        --height 1.80 \\
        --frame_idx 150
"""

import argparse
import json
import os
import sys
import glob
import re

import cv2
import numpy as np

# Add repo root for util import (script lives in postprocessing/);
# sibling evaluate_calibration is found automatically since Python adds the
# script's own directory to sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core import load_poses
from evaluate_calibration import export_to_toml

# Keypoint indices in Halpe26 (used with RTMPose)
HALPE26_HEAD = 17
HALPE26_L_HEEL = 24
HALPE26_R_HEEL = 25
HALPE26_L_BIG_TOE = 20
HALPE26_R_BIG_TOE = 21
HALPE26_L_SMALL_TOE = 22
HALPE26_R_SMALL_TOE = 23

# Keypoint indices in Calib26 / MeTRAbs format
CALIB26_HEAD = 0
CALIB26_L_HEEL = 22
CALIB26_R_HEEL = 23
CALIB26_L_TOE = 24
CALIB26_R_TOE = 25
CALIB26_L_FOO = 20
CALIB26_R_FOO = 21

# Keypoint indices in bml_movi_87 (full MeTRAbs skeleton)
BML87_HEAD = 67
BML87_L_HEEL = 21
BML87_R_HEEL = 52
BML87_L_TOE = 23
BML87_R_TOE = 54
BML87_L_FOO = 78
BML87_R_FOO = 86
BML87_L_FIFTHMET = 22
BML87_R_FIFTHMET = 53

def get_3d_keypoint(p2d_all, s2d_all, K, R_w2c, t_w2c, frame_idx, joint_idx, conf_threshold=0.5):
    """Triangulates a single 3D keypoint for a specific frame."""
    C = p2d_all.shape[0]
    p2d_frame = p2d_all[:, frame_idx, joint_idx, :]
    s2d_frame = s2d_all[:, frame_idx, joint_idx]

    vis_mask = s2d_frame > conf_threshold # Use a confidence threshold
    if np.sum(vis_mask) < 2:
        return None

    pts_vis = p2d_frame[vis_mask]
    Ps_all = [K[c] @ np.hstack([R_w2c[c], t_w2c[c].reshape(3, 1)]) for c in range(C)]
    Ps_vis = [Ps_all[c] for c, is_vis in enumerate(vis_mask) if is_vis]

    rows = []
    for pt, P in zip(pts_vis, Ps_vis):
        x, y = pt
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]

    return Xh[:3] / Xh[3] if abs(Xh[3]) > 1e-10 else None


def main():
    parser = argparse.ArgumentParser(description="Scales and reorients a scene to metric units.")
    parser.add_argument("--prefix", required=True, help="Path to the session folder")
    parser.add_argument("--calib", required=True, help="Name of the best calibration JSON file")
    parser.add_argument("--height", required=True, type=float, help="Real-world height of the person in meters")
    parser.add_argument("--frame_idx", required=True, type=int, help="Index of a frame where the person is standing straight")
    parser.add_argument("--subset", default="noise_1_0", help="Subset folder name")
    parser.add_argument("--input_toml", default=None, help="Path to the input TOML file template")
    parser.add_argument("--export_toml", default=None, help="Path to save the final calibrated TOML file")
    parser.add_argument("--video_dir", default=None, help="Path to original videos (needed for TOML export)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for 2D keypoints")
    parser.add_argument("--pose_engine", default="rtmpose", choices=["rtmpose", "metrabs"],
                        help="Pose engine used: determines joint format for scaling")
    args = parser.parse_args()

    # --- Select joint format based on pose engine ---
    # Auto-detect: check if 2d_joint has 87 joints (full bml_movi_87)
    _detect_dir = os.path.join(args.prefix, args.subset, "2d_joint")
    _n_joints_detected = 0
    if os.path.isdir(_detect_dir):
        _sample = sorted(glob.glob(os.path.join(_detect_dir, "*.json")))
        if _sample:
            import json as _json
            with open(_sample[0]) as _f:
                _d = _json.load(_f)
                if _d["data"]:
                    _n_joints_detected = len(_d["data"][0]["skeleton"][0]["score"])

    if args.pose_engine == "metrabs" and _n_joints_detected == 87:
        joint_dir = os.path.join(args.prefix, args.subset, "2d_joint")
        HEAD_IDX = BML87_HEAD
        L_HEEL_IDX = BML87_L_HEEL
        R_HEEL_IDX = BML87_R_HEEL
        foot_kp_indices = [BML87_L_HEEL, BML87_R_HEEL, BML87_L_TOE, BML87_R_TOE,
                           BML87_L_FOO, BML87_R_FOO, BML87_L_FIFTHMET, BML87_R_FIFTHMET]
        print(f"  Using MeTRAbs bml_movi_87 joints for scaling ({_n_joints_detected} joints)")
    elif args.pose_engine == "metrabs":
        joint_dir = os.path.join(args.prefix, args.subset, "2d_joint")
        HEAD_IDX = CALIB26_HEAD
        L_HEEL_IDX = CALIB26_L_HEEL
        R_HEEL_IDX = CALIB26_R_HEEL
        foot_kp_indices = [CALIB26_L_HEEL, CALIB26_R_HEEL, CALIB26_L_TOE, CALIB26_R_TOE,
                           CALIB26_L_FOO, CALIB26_R_FOO]
        print(f"  Using MeTRAbs calib26 joints for scaling ({_n_joints_detected} joints)")
    else:
        joint_dir = os.path.join(args.prefix, args.subset, "2d_joint_halpe26")
        HEAD_IDX = HALPE26_HEAD
        L_HEEL_IDX = HALPE26_L_HEEL
        R_HEEL_IDX = HALPE26_R_HEEL
        foot_kp_indices = [HALPE26_L_HEEL, HALPE26_R_HEEL, HALPE26_L_BIG_TOE,
                           HALPE26_R_BIG_TOE, HALPE26_L_SMALL_TOE, HALPE26_R_SMALL_TOE]
    halpe26_dir = joint_dir  # used below for loading poses
    calib_json_path = os.path.join(args.prefix, "results", f"{args.calib}.json")

    with open(calib_json_path, 'r') as f:
        calib_data = json.load(f)
    
    CAMID = calib_data['CAMID']
    K = np.array(calib_data['K'])
    R_w2c_orig = np.array(calib_data['R_w2c'])
    t_w2c_orig = np.array(calib_data['t_w2c'])

    base_name = os.path.basename(os.path.normpath(args.prefix))
    match = re.search(r'A(\d+)_P(\d+)_G(\d+)', base_name)
    if match:
        aid, pid, gid = int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        # Fallback for folder names that don't follow the Axxx_Pxxx_Gxxx convention
        print(f"WARNING: Output directory '{base_name}' does not match Axxx_Pxxx_Gxxx format. Using default IDs (AID=1, PID=1, GID=1).")
        aid, pid, gid = 1, 1, 1

    p2d_list, s2d_list = [], []
    min_frames = float('inf')
    for cid in CAMID:
        fpath = os.path.join(halpe26_dir, f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json")
        if not os.path.exists(fpath):
            print(f"ERROR: Halpe26 pose file not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        frames, p2d, s2d = load_poses(fpath)
        min_frames = min(min_frames, len(frames))
        p2d_list.append(p2d)
        s2d_list.append(s2d)

    num_joints = p2d_list[0].shape[1] // 2
    p2d_all = np.array([p[:min_frames].reshape(min_frames, num_joints, 2) for p in p2d_list])
    s2d_all = np.array([s[:min_frames].reshape(min_frames, num_joints) for s in s2d_list])

    # --- Triangulate Foot and Head Keypoints ---
    print(f"Triangulating keypoints for frame {args.frame_idx}...")
    foot_points_3d = [get_3d_keypoint(p2d_all, s2d_all, K, R_w2c_orig, t_w2c_orig, args.frame_idx, idx, args.conf_threshold) for idx in foot_kp_indices]
    foot_points_3d = [p for p in foot_points_3d if p is not None]

    if len(foot_points_3d) < 3:
        print(f"ERROR: Not enough foot keypoints ({len(foot_points_3d)}) to define a plane. Try a different frame.", file=sys.stderr)
        sys.exit(1)

    head_3d = get_3d_keypoint(p2d_all, s2d_all, K, R_w2c_orig, t_w2c_orig, args.frame_idx, HEAD_IDX, args.conf_threshold)
    if head_3d is None:
        print("ERROR: Could not triangulate head. Cannot calculate scale.", file=sys.stderr)
        sys.exit(1)

    # --- 1. Define New Coordinate System ---
    ground_centroid = np.mean(foot_points_3d, axis=0)
    
    # The vertical axis (Y-axis) is defined by the body's vertical vector
    # (from head to foot centroid). In OpenCV convention, Y points DOWN.
    y_axis_temp = ground_centroid - head_3d
    y_axis = y_axis_temp / np.linalg.norm(y_axis_temp)

    # X-axis can be defined by the vector between heels
    l_heel_3d = get_3d_keypoint(p2d_all, s2d_all, K, R_w2c_orig, t_w2c_orig, args.frame_idx, L_HEEL_IDX, args.conf_threshold)
    r_heel_3d = get_3d_keypoint(p2d_all, s2d_all, K, R_w2c_orig, t_w2c_orig, args.frame_idx, R_HEEL_IDX, args.conf_threshold)
    
    if l_heel_3d is not None and r_heel_3d is not None:
        # X-axis should point to the right (Left Heel -> Right Heel to fix mirror effect)
        x_axis_temp = l_heel_3d - r_heel_3d
        heels_center = (l_heel_3d + r_heel_3d) / 2.0
    else:
        x_axis_temp = np.array([1.0, 0.0, 0.0])
        heels_center = ground_centroid

    x_axis = x_axis_temp - np.dot(x_axis_temp, y_axis) * y_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis_temp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = x_axis_temp - np.dot(x_axis_temp, y_axis) * y_axis

    x_axis = x_axis / np.linalg.norm(x_axis)

    # Z-axis is the cross product
    z_axis = np.cross(x_axis, y_axis)

    # Origin (0,0,0) is placed at the center of the heels (for X,Z axes),
    # and aligned with the lowest point of all foot points (for Y axis).
    floor_y_proj = max(np.dot(p, y_axis) for p in foot_points_3d)
    origin = heels_center + (floor_y_proj - np.dot(heels_center, y_axis)) * y_axis

    # --- 2. Create Transformation Matrix ---
    R_transform = np.vstack([x_axis, y_axis, z_axis])

    # --- 3. Transform Cameras ---
    R_w2c_new, t_w2c_new = [], []
    for R, t in zip(R_w2c_orig, t_w2c_orig):
        # Direct algebraic transformation without unstable matrix inversions
        R_new = R @ R_transform.T
        t_new = R @ origin.reshape(3, 1) + t.reshape(3, 1)

        R_w2c_new.append(R_new)
        t_w2c_new.append(t_new)

    R_w2c_new, t_w2c_new = np.array(R_w2c_new), np.array(t_w2c_new)

    # --- 4. Calculate and Apply Scale ---
    head_3d_new = (R_transform @ head_3d) - (R_transform @ origin)
    
    # The new Y coordinate of the head is negative (Y points down). Height is absolute value.
    measured_height = abs(head_3d_new[1])
    
    scale_factor = args.height / measured_height
    print(f"Calculated scale factor: {scale_factor:.4f}")

    t_w2c_scaled = t_w2c_new * scale_factor

    # --- 5. Save Final Calibrated Files ---
    output_calib_path = os.path.join(args.prefix, "results", f"{args.calib}_oriented_scaled.json")
    out_data = {
        "CAMID": CAMID, "K": K.tolist(), "R_w2c": R_w2c_new.tolist(), "t_w2c": t_w2c_scaled.tolist(),
    }
    with open(output_calib_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved final oriented and scaled calibration to: {output_calib_path}")

    if args.export_toml:
        if not args.video_dir:
            parser.error("--video_dir is required for TOML export.")
        video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.MP4")) + glob.glob(os.path.join(args.video_dir, "*.mp4")))
        cam_names = [os.path.splitext(os.path.basename(f))[0] for f in video_files]
        export_to_toml(args.input_toml, args.export_toml, R_w2c_new, t_w2c_scaled, cam_names)

if __name__ == "__main__":
    main()
