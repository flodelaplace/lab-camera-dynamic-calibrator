"""
evaluate_calibration.py
-----------------------
Calculates the Mean Reprojection Error (MRE) for a given calibration file,
providing both a global score and a per-camera breakdown.

It can also generate visualizations of the best and worst frames for each camera,
and export the final calibration to a TOML file.

Usage:
    # Calculate MRE and generate visualizations
    python evaluate_calibration.py \
        --prefix     ./data/A001_P002_G001 \
        --calib      linear_1_0_ba \
        --video_dir  /path/to/videos \
        --visualize

    # Export the best calibration to a TOML file
    python evaluate_calibration.py \
        --prefix       ./data/A001_P002_G001 \
        --calib        linear_1_0_ba \
        --input_toml   /path/to/original.toml \
        --export_toml  /path/to/output.toml
"""

import argparse
import json
import os
import sys
import glob
import re

import cv2
import numpy as np
import yaml

try:
    import tomli as tomllib
except ImportError:
    print("ERROR: 'tomli' is required for TOML parsing. Please run: pip install tomli", file=sys.stderr)
    tomllib = None


# Add repo root for util import (script lives in postprocessing/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from util import load_poses, load_eldersim_camera

def triangulate_skeleton(p2d_all, s2d_all, K, R_w2c, t_w2c, conf_threshold=0.5):
    C, N, J, _ = p2d_all.shape
    X3d = np.full((N, J, 3), np.nan)
    Ps_all = [K[c] @ np.hstack([R_w2c[c], t_w2c[c].reshape(3, 1)]) for c in range(C)]

    for n in range(N):
        for j in range(J):
            vis_mask = s2d_all[:, n, j] > conf_threshold
            if np.sum(vis_mask) < 2:
                continue
            
            pts_vis = p2d_all[vis_mask, n, j]
            Ps_vis = [Ps_all[c] for c, is_vis in enumerate(vis_mask) if is_vis]

            rows = []
            for pt, P in zip(pts_vis, Ps_vis):
                x, y = pt
                rows.append(x * P[2] - P[0])
                rows.append(y * P[2] - P[1])
            A = np.array(rows)
            _, _, Vt = np.linalg.svd(A)
            Xh = Vt[-1]
            if abs(Xh[3]) > 1e-10:
                X3d[n, j] = Xh[:3] / Xh[3]
    return X3d

def reproject_points(X3d_world, K, R_w2c, t_w2c):
    C = len(K)
    N, J, _ = X3d_world.shape
    p2d_reprojected_all = np.full((C, N, J, 2), np.nan)

    for c in range(C):
        X3d_contiguous = np.ascontiguousarray(X3d_world.reshape(-1, 3))
        nan_mask = ~np.isnan(X3d_contiguous).any(axis=1)
        
        if np.any(nan_mask):
            rvec, _ = cv2.Rodrigues(R_w2c[c])
            projected, _ = cv2.projectPoints(X3d_contiguous[nan_mask], rvec, t_w2c[c], K[c], None)
            
            temp_reprojected = np.full((N * J, 2), np.nan)
            temp_reprojected[nan_mask] = projected.reshape(-1, 2)
            p2d_reprojected_all[c] = temp_reprojected.reshape(N, J, 2)
            
    return p2d_reprojected_all

def draw_visualization(image, p2d_original, p2d_reprojected, s2d, title, conf_threshold=0.5):
    vis_img = image.copy()
    valid_mask = s2d > conf_threshold

    for j in range(p2d_original.shape[0]):
        if valid_mask[j]:
            if np.any(np.isnan(p2d_reprojected[j])) or np.any(np.isinf(p2d_reprojected[j])):
                continue
            orig_pt = tuple(p2d_original[j].astype(int))
            reproj_pt = tuple(p2d_reprojected[j].astype(int))

            cv2.line(vis_img, orig_pt, reproj_pt, (0, 255, 255), 1) # Yellow line for error
            cv2.circle(vis_img, orig_pt, 4, (255, 0, 0), -1)      # Blue circle for original
            cv2.circle(vis_img, reproj_pt, 4, (0, 0, 255), -1)    # Red circle for reprojected

    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return vis_img

def export_to_toml(input_toml_path, output_toml_path, R_w2c, t_w2c, cam_names):
    with open(input_toml_path, "r") as f:
        lines = f.readlines()

    output_lines = []
    current_cam_index = -1
    cam_name_pattern = re.compile(r'\[([^]]+)\]')

    for line in lines:
        match = cam_name_pattern.match(line)
        if match:
            cam_name = match.group(1)
            try:
                current_cam_index = cam_names.index(cam_name)
            except ValueError:
                current_cam_index = -1
            output_lines.append(line)
        elif current_cam_index != -1 and line.strip().startswith('rotation'):
            rvec, _ = cv2.Rodrigues(R_w2c[current_cam_index])
            r_str = ', '.join(map(str, rvec.flatten()))
            output_lines.append(f"rotation = [{r_str}]\n")
        elif current_cam_index != -1 and line.strip().startswith('translation'):
            t_str = ', '.join(map(str, t_w2c[current_cam_index].flatten()))
            output_lines.append(f"translation = [{t_str}]\n")
        else:
            output_lines.append(line)

    with open(output_toml_path, "w") as f:
        f.writelines(output_lines)

    print(f"\nSuccessfully exported final calibration to: {output_toml_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculates MRE and handles calibration file operations.")
    parser.add_argument("--prefix", required=True, help="Path to the session folder")
    parser.add_argument("--calib", required=True, help="Name of the calibration JSON file")
    parser.add_argument("--subset", default="noise_1_0", help="Subset folder name")
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true", help="Generate visualization images")
    parser.add_argument("--video_dir", default=None, help="Path to original videos (for --visualize)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index to align with original video")
    # TOML export arguments
    parser.add_argument("--input_toml", default=None, help="Path to the input TOML file to use as a template")
    parser.add_argument("--export_toml", default=None, help="Path to save the final calibrated TOML file")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for 2D keypoints")
    args = parser.parse_args()

    if args.visualize and not args.video_dir:
        parser.error("--video_dir is required when --visualize is set.")
    if args.export_toml and not args.input_toml:
        parser.error("--input_toml is required when --export_toml is set.")

    # --- Setup Paths ---
    subset_dir = os.path.join(args.prefix, args.subset)
    calib_json_path = os.path.join(args.prefix, "results", f"{args.calib}.json")
    if not os.path.exists(calib_json_path):
        print(f"ERROR: Calibration file not found: {calib_json_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load Data ---
    CAMID, K, R_w2c, t_w2c, _dist = load_eldersim_camera(calib_json_path)
    camera_ids = CAMID

    # --- TOML Export Logic ---
    if args.export_toml:
        if not args.video_dir: # We need video_dir to get the cam_names in order
             parser.error("--video_dir is required for TOML export to determine camera name order.")
        video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.MP4")) + glob.glob(os.path.join(args.video_dir, "*.mp4")))
        cam_names = [os.path.splitext(os.path.basename(f))[0] for f in video_files]
        export_to_toml(args.input_toml, args.export_toml, R_w2c, t_w2c, cam_names)
        sys.exit(0) # Exit after exporting

    # --- MRE Calculation and Visualization Logic ---
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
    for cid in camera_ids:
        fpath = os.path.join(subset_dir, "2d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json")
        if not os.path.exists(fpath):
            print(f"ERROR: 2D pose file not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        frames, p2d, s2d = load_poses(fpath)
        min_frames = min(min_frames, len(frames))
        p2d_list.append(p2d)
        s2d_list.append(s2d)

    num_joints = p2d_list[0].shape[1] // 2
    p2d_all = np.array([p[:min_frames].reshape(min_frames, num_joints, 2) for p in p2d_list])
    s2d_all = np.array([s[:min_frames].reshape(min_frames, num_joints) for s in s2d_list])

    X3d_world = triangulate_skeleton(p2d_all, s2d_all, K, R_w2c, t_w2c, args.conf_threshold)
    p2d_reprojected_all = reproject_points(X3d_world, K, R_w2c, t_w2c)
    all_errors = np.linalg.norm(p2d_all - p2d_reprojected_all, axis=-1)
    valid_mask = (s2d_all > args.conf_threshold) & ~np.isnan(all_errors)

    print(f"\nEvaluating calibration: {args.calib}.json")
    global_mre = np.mean(all_errors[valid_mask])
    print(f"\n  -> Global MRE: {global_mre:.3f} pixels")
    print("\n  -> Per-camera MRE:")
    for c, cam_id in enumerate(CAMID):
        cam_mask = valid_mask[c]
        cam_mre = np.mean(all_errors[c][cam_mask]) if np.any(cam_mask) else -1
        print(f"     - Camera {cam_id}: {cam_mre:.3f} pixels")
    print("-" * 40)

    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = os.path.join(args.prefix, "results", "MRE_visualizations", args.calib)
        os.makedirs(vis_dir, exist_ok=True)
        video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.MP4")) + glob.glob(os.path.join(args.video_dir, "*.mp4")))

        for c, cam_id in enumerate(CAMID):
            frame_errors = np.nanmean(np.where(valid_mask[c], all_errors[c], np.nan), axis=1)
            if np.all(np.isnan(frame_errors)):
                print(f"  Skipping Cam {cam_id}: No valid frames to visualize.")
                continue

            best_frame_idx, worst_frame_idx = np.nanargmin(frame_errors), np.nanargmax(frame_errors)
            video_path = video_files[c]
            cap = cv2.VideoCapture(video_path)

            for frame_type, frame_idx in [("best", best_frame_idx), ("worst", worst_frame_idx)]:
                video_frame_idx = frame_idx + args.start_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, img = cap.read()
                if ret:
                    title = f"Cam {cam_id} - {frame_type.capitalize()} Frame (MRE: {frame_errors[frame_idx]:.2f}px)"
                    vis_img = draw_visualization(img, p2d_all[c, frame_idx], p2d_reprojected_all[c, frame_idx], s2d_all[c, frame_idx], title, args.conf_threshold)
                    out_path = os.path.join(vis_dir, f"cam{cam_id}_{frame_type}.png")
                    cv2.imwrite(out_path, vis_img)
                    print(f"  Saved: {out_path}")
            cap.release()

if __name__ == "__main__":
    main()
