#!/usr/bin/env python3
"""Detect per-camera outlier frames after linear calibration.

For each camera, computes the mean per-frame reprojection error and flags
frames whose error exceeds BOTH ``--abs_px`` AND ``--x_median * median``.
Outliers are appended to the per-video sidecar ``<video>.dropped.json`` and
their scores are zeroed in the saved 2D/3D pose JSONs so an immediate
re-run of the linear calibration sees them as drops without re-running
MeTRAbs.

Prints ``NEW_DROPS=<n>`` on the last line for shell consumption.

Usage:
    python scripts/detect_outlier_frames.py \\
        --prefix ./output/calib_remi_unitapa \\
        --subset noise_1_0 --aid 1 --pid 1 --gid 1 \\
        --calib linear_1_0 \\
        --video_dir ./input/calib_remi_unitapa \\
        --abs_px 50 --x_median 5
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from postprocessing.evaluate_calibration import triangulate_skeleton, reproject_points
from core import load_poses, load_eldersim_camera


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--subset", default="noise_1_0")
    p.add_argument("--aid", type=int, default=1)
    p.add_argument("--pid", type=int, default=1)
    p.add_argument("--gid", type=int, default=1)
    p.add_argument("--calib", default="linear_1_0")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--abs_px", type=float, default=50.0)
    p.add_argument("--x_median", type=float, default=5.0)
    p.add_argument("--conf_threshold", type=float, default=0.5)
    return p.parse_args()


def load_camera_poses(prefix, subset, aid, pid, gid, n_cams):
    """Load per-camera 2D/3D poses. Frame indices are assumed identical across cams."""
    p2d_all, s2d_all = [], []
    frame_indices = None
    for cid in range(1, n_cams + 1):
        fname = f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"
        f2d, p2d, s2d = load_poses(os.path.join(prefix, subset, "2d_joint", fname))
        if frame_indices is None:
            frame_indices = [int(f) for f in f2d]
        n_joints = s2d.shape[1]
        p2d_all.append(p2d.reshape(-1, n_joints, 2))
        s2d_all.append(s2d)
    return frame_indices, np.array(p2d_all), np.array(s2d_all)


def per_frame_reproj_errors(p2d, s2d, X3d_world, K, R_w2c, t_w2c, conf_threshold):
    """Mean reprojection error per camera per frame. Returns (C, N), NaN if no visible joints."""
    p2d_reproj = reproject_points(X3d_world, K, R_w2c, t_w2c)
    err = np.linalg.norm(p2d_reproj - p2d, axis=-1)
    valid = (s2d > conf_threshold) & ~np.isnan(err)
    with np.errstate(invalid='ignore', all='ignore'):
        err_masked = np.where(valid, err, np.nan)
        return np.nanmean(err_masked, axis=2)


def detect_outliers(frame_errors, abs_px, x_median):
    """Per-camera outlier indices into the frame_indices array."""
    C, _ = frame_errors.shape
    outliers = []
    for c in range(C):
        errs = frame_errors[c]
        with np.errstate(invalid='ignore', all='ignore'):
            med = np.nanmedian(errs)
        if np.isnan(med):
            outliers.append(set())
            continue
        thr = max(abs_px, x_median * med)
        bad = np.where((errs > thr) & ~np.isnan(errs))[0]
        outliers.append({int(i) for i in bad})
    return outliers


def update_sidecar(video_path, new_indices):
    """Merge ``new_indices`` (absolute frame numbers) into ``<video>.dropped.json``."""
    sidecar = os.path.splitext(video_path)[0] + '.dropped.json'
    existing = []
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            existing = json.load(f).get('dropped_frame_indices', [])
    existing_set = {int(i) for i in existing}
    merged = existing_set | new_indices
    added = len(merged) - len(existing_set)
    if added == 0:
        return 0
    with open(sidecar, 'w') as f:
        json.dump({'dropped_frame_indices': sorted(merged)}, f, indent=2)
    return added


def zero_scores_in_json(json_path, dropped_frames):
    """Set score and pose to zeros for entries whose frame_index is in dropped_frames."""
    with open(json_path) as f:
        d = json.load(f)
    n_changed = 0
    for entry in d['data']:
        if int(entry['frame_index']) in dropped_frames:
            for sk in entry['skeleton']:
                n_joints = len(sk['score'])
                pose_dim = len(sk['pose']) // n_joints if n_joints else 0
                sk['score'] = [0.0] * n_joints
                sk['pose'] = [0.0] * (n_joints * pose_dim)
                n_changed += 1
    if n_changed > 0:
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=2)
    return n_changed


def main():
    args = parse_args()

    calib_path = os.path.join(args.prefix, "results", f"{args.calib}.json")
    CAMID, K, R_w2c, t_w2c, _ = load_eldersim_camera(calib_path)
    n_cams = len(CAMID)

    frame_indices, p2d, s2d = load_camera_poses(
        args.prefix, args.subset, args.aid, args.pid, args.gid, n_cams
    )

    X3d = triangulate_skeleton(p2d, s2d, K, R_w2c, t_w2c, args.conf_threshold)
    frame_errors = per_frame_reproj_errors(p2d, s2d, X3d, K, R_w2c, t_w2c, args.conf_threshold)
    outliers = detect_outliers(frame_errors, args.abs_px, args.x_median)

    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.video_dir, ext)))
    video_files = sorted(video_files)
    if len(video_files) != n_cams:
        print(f"ERROR: found {len(video_files)} videos in {args.video_dir} "
              f"but calib has {n_cams} cams", file=sys.stderr)
        sys.exit(1)

    print(f"Thresholds: error > {args.abs_px}px AND error > {args.x_median}x median")
    total_added = 0
    for c in range(n_cams):
        cid = int(CAMID[c])
        with np.errstate(invalid='ignore', all='ignore'):
            med = float(np.nanmedian(frame_errors[c]))
            mx = float(np.nanmax(frame_errors[c])) if np.any(~np.isnan(frame_errors[c])) else float('nan')

        bad_idx = outliers[c]
        if not bad_idx:
            print(f"  Cam {cid}: 0 outliers (median {med:.1f}px, max {mx:.1f}px)")
            continue

        bad_frames = {frame_indices[i] for i in bad_idx}
        worst = float(np.nanmax([frame_errors[c, i] for i in bad_idx]))
        print(f"  Cam {cid}: {len(bad_frames)} outliers (median {med:.1f}px, "
              f"worst-outlier {worst:.1f}px)")

        added = update_sidecar(video_files[c], bad_frames)
        total_added += added

        fname = f"A{args.aid:03d}_P{args.pid:03d}_G{args.gid:03d}_C{cid:03d}.json"
        for sub in ("2d_joint", "3d_joint", "2d_joint_halpe26"):
            jp = os.path.join(args.prefix, args.subset, sub, fname)
            if os.path.exists(jp):
                zero_scores_in_json(jp, bad_frames)

    print(f"NEW_DROPS={total_added}")


if __name__ == "__main__":
    main()
