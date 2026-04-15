"""
visualize_results.py
---------------------
Visualisation frame par frame du squelette 3D triangulé + position des caméras
à partir des résultats de calibration extrinsèque.

Usage:
    python visualize_results.py \
        --prefix   ./data/A001_P001_G001 \
        --subset   noise_1_0 \
        --calib    linear_1_0_ba \
        --dataset  MyDataset \
        --output   ./data/A001_P001_G001/results/camera/visu_3d.gif

    # Pour un MP4 au lieu d'un GIF :
        --output   ./data/A001_P001_G001/results/camera/visu_3d.mp4
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d
import yaml

# Add repo root for util import (script lives in postprocessing/) and locate VideoPose3D
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

vp3d_path = os.path.join(_REPO_ROOT, "third_party", "VideoPose3D")
if vp3d_path not in sys.path:
    sys.path.insert(0, vp3d_path)

from util import load_poses, load_eldersim_camera

# ── squelette OpenPose-25 ─────────────────────────────────────────────────────
OPENPOSE_SKELETON = (
    (1, 8), (1, 2), (1, 5), (0, 15), (0, 16),
    (15, 17), (16, 18), (1, 0),
    (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (8, 12),
    (9, 10), (12, 13),
    (10, 11), (13, 14),
    (11, 22), (11, 24), (22, 23),
    (14, 19), (14, 21), (19, 20),
)
OP_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

# ── squelette MeTRAbs calib-26 ────────────────────────────────────────────────
METRABS_SKELETON = (
    # Spine
    (5, 4), (4, 2), (2, 1), (1, 0),
    # Torso
    (3, 2), (6, 7), (14, 15),
    # Shoulders
    (2, 6), (2, 7),
    # Hips
    (4, 14), (4, 15),
    # Left arm
    (6, 8), (8, 10), (10, 12),
    # Right arm
    (7, 9), (9, 11), (11, 13),
    # Left leg
    (14, 16), (16, 18), (18, 20),
    # Right leg
    (15, 17), (17, 19), (19, 21),
    # Feet
    (18, 22), (22, 24),
    (19, 23), (23, 25),
)
METRABS_NAMES = [
    "head", "backneck", "thor", "sternum", "pelv", "mhip",
    "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
    "lhan", "rhan", "lhip", "rhip", "lkne", "rkne",
    "lank", "rank", "lfoo", "rfoo", "lhee", "rhee", "ltoe", "rtoe",
]


def export_to_trc(X3d_world, output_path, fps=30.0):
    """
    Exporte les coordonnées 3D au format .trc (pour OpenSim / Mokka)
    """
    import os
    N, J, _ = X3d_world.shape
    if J == 87:
        from util import BML87_KEY
        joint_names = [k for k, v in sorted(BML87_KEY.items(), key=lambda x: x[1])]
    elif J == 26:
        joint_names = METRABS_NAMES
    else:
        joint_names = OP_NAMES[:J]
    for i in range(len(joint_names), J):
        joint_names.append(f"Joint_{i}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(output_path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{N}\t{J}\tm\t{fps:.2f}\t1\t{N}\n")
        f.write("Frame#\tTime\t")
        for name in joint_names:
            f.write(f"{name}\t\t\t")
        f.write("\n\t\t")
        for i in range(1, J + 1):
            f.write(f"X{i}\tY{i}\tZ{i}\t")
        f.write("\n\n")
        for n in range(N):
            time = n / fps
            f.write(f"{n + 1}\t{time:.5f}\t")
            for j in range(J):
                pt = X3d_world[n, j]
                if np.isnan(pt).any():
                    f.write("\t\t\t")
                else:
                    f.write(f"{pt[0]:.5f}\t{pt[1]:.5f}\t{pt[2]:.5f}\t")
            f.write("\n")
    print(f"Saved TRC file to: {output_path}")

BONE_COLOR  = "#f94e3e"
CAM_COLOR   = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
               "#00BCD4", "#FFEB3B", "#795548", "#607D8B"]


def triangulate_skeleton(p2d_all, s2d_all, K, R_w2c, t_w2c, conf_threshold=0.5):
    """
    Triangule les joints 2D de toutes les caméras pour obtenir un squelette
    dans le repère monde. Utilise toutes les caméras visibles via DLT.
    p2d_all : (C, N, J, 2)
    s2d_all : (C, N, J)
    Retourne : (N, J, 3)
    """
    C, N, J, _ = p2d_all.shape
    X3d = np.full((N, J, 3), np.nan)

    Ps_all = [K[c] @ np.hstack([R_w2c[c], t_w2c[c].reshape(3, 1)]) for c in range(C)]

    for n in range(N):
        for j in range(J):
            vis = s2d_all[:, n, j] > conf_threshold
            if np.sum(vis) < 2:
                continue
            
            pts_vis = p2d_all[vis, n, j]
            Ps_vis  = [Ps_all[c] for c, is_vis in enumerate(vis) if is_vis]

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


def draw_camera(ax, R_w2c, t_w2c, color, label, scale=0.15):
    """Dessine une pyramide représentant une caméra dans le repère monde."""
    C = (-R_w2c.T @ t_w2c.reshape(3, 1)).flatten()
    
    w, h, f = 0.8, 0.6, 1.0
    corners_cam = np.array([[w, h, f], [-w, h, f], [-w, -h, f], [w, -h, f]]) * scale
    corners_world = (R_w2c.T @ corners_cam.T).T + C

    # Mapping pour l'affichage Matplotlib : (X_cv, Y_cv, Z_cv) -> (X, Z, -Y)
    C_plot = [C[0], C[2], -C[1]]
    corners_plot = np.empty_like(corners_world)
    corners_plot[:, 0] = corners_world[:, 0]
    corners_plot[:, 1] = corners_world[:, 2]
    corners_plot[:, 2] = -corners_world[:, 1]

    for corner in corners_plot:
        ax.plot([C_plot[0], corner[0]], [C_plot[1], corner[1]], [C_plot[2], corner[2]],
                color=color, linewidth=1.5, alpha=0.8)
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners_plot[i, 0], corners_plot[j, 0]],
                [corners_plot[i, 1], corners_plot[j, 1]],
                [corners_plot[i, 2], corners_plot[j, 2]],
                color=color, linewidth=1.5, alpha=0.8)

    ax.scatter(*C_plot, c=color, s=60, zorder=10, marker="o", depthshade=False)
    ax.text(C_plot[0], C_plot[1], C_plot[2] + 0.1, f"  {label}", color=color, fontsize=9, fontweight="bold", ha='center')


def make_animation(X3d_world, R_w2c, t_w2c, output_path, fps=15, step=1):
    """
    Génère un GIF/MP4 animé avec le squelette triangulé + les caméras.
    """
    N = X3d_world.shape[0]
    frames_to_render = list(range(0, N, step))

    cam_positions = np.array([(-R.T @ t.reshape(3, 1)).flatten() for R, t in zip(R_w2c, t_w2c)])

    valid = X3d_world[~np.isnan(X3d_world).any(axis=-1)].reshape(-1, 3)
    
    # Bounding box pour le squelette (robuste au bruit via les percentiles)
    if len(valid) > 0:
        valid_plot = np.copy(valid)
        valid_plot[:, 0] = valid[:, 0]
        valid_plot[:, 1] = valid[:, 2]
        valid_plot[:, 2] = -valid[:, 1]
        skel_min = np.percentile(valid_plot, 2, axis=0)
        skel_max = np.percentile(valid_plot, 98, axis=0)
    else:
        skel_min = np.array([np.inf, np.inf, np.inf])
        skel_max = np.array([-np.inf, -np.inf, -np.inf])
        
    # Bounding box pour les caméras (inclusions strictes avec min/max)
    cam_plot = np.copy(cam_positions)
    cam_plot[:, 0] = cam_positions[:, 0]
    cam_plot[:, 1] = cam_positions[:, 2]
    cam_plot[:, 2] = -cam_positions[:, 1]
    cam_min = np.min(cam_plot, axis=0)
    cam_max = np.max(cam_plot, axis=0)

    pad = 0.2  # Marge réduite pour maximiser le zoom
    xmin, ymin, zmin = np.minimum(skel_min, cam_min) - pad
    xmax, ymax, zmax = np.maximum(skel_max, cam_max) + pad
    
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2

    cam_colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800",
                  "#9C27B0", "#00BCD4", "#FFEB3B", "#795548"]
    n_cams = len(R_w2c)
    scale = max_range * 0.18

    fig = plt.figure(figsize=(12, 10), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95) # Supprime les bordures noires inutiles
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    method_labels = {
        "linear_1_0":     "Linear",
        "linear_1_0_ba":  "Linear + Bundle Adjustment ⭐",
        "ransac_1_0":     "RANSAC",
        "ransac_1_0_ba":  "RANSAC + Bundle Adjustment",
    }
    calib_name = os.path.splitext(os.path.basename(output_path))[0].replace("visu_3d_", "")
    method_label = method_labels.get(calib_name, calib_name)

    def draw_frame(fc):
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_title(f"{method_label}  —  Frame {fc+1}/{N}  —  {n_cams} cameras",
                     color="white", fontsize=10)
        
        # Set labels for X, Z (floor), Y (vertical)
        ax.set_xlabel("X", color="gray"); ax.set_ylabel("Z", color="gray")
        ax.set_zlabel("Y (vertical)", color="gray")
        ax.tick_params(colors="gray")

        ax.set_xlim(cx - max_range, cx + max_range)
        ax.set_ylim(cy - max_range, cy + max_range)
        ax.set_zlim(cz - max_range, cz + max_range)
        
        ax.view_init(elev=20, azim=-60)

        pts = X3d_world[fc]
        pts_plot = np.copy(pts)
        pts_plot[:, 0] = pts[:, 0]
        pts_plot[:, 1] = pts[:, 2]
        pts_plot[:, 2] = -pts[:, 1]

        valid_mask = ~np.isnan(pts_plot).any(axis=1)
        ax.scatter(pts_plot[valid_mask, 0], pts_plot[valid_mask, 1], pts_plot[valid_mask, 2],
                   c="white", s=30, zorder=5, depthshade=False) # Points plus gros
        skeleton = (METRABS_SKELETON if X3d_world.shape[1] == 26
                    else OPENPOSE_SKELETON)
        if X3d_world.shape[1] == 87:
            from util import BML87_BONE
            skeleton = [(int(b[0]), int(b[1])) for b in BML87_BONE]
        for j0, j1 in skeleton:
            p0, p1 = pts_plot[j0], pts_plot[j1]
            if not (np.isnan(p0).any() or np.isnan(p1).any()):
                line = art3d.Line3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                                    color=BONE_COLOR, linewidth=3, alpha=1.0) # Traits plus épais
                ax.add_line(line)

        # Draw cameras
        for i, (R, t) in enumerate(zip(R_w2c, t_w2c)):
            color = cam_colors[i % len(cam_colors)]
            draw_camera(ax, R, t, color=color, label=f"Cam{i+1}", scale=scale)

    print(f"Rendering {len(frames_to_render)} frames...")
    ani = animation.FuncAnimation(fig, draw_frame,
                                  frames=frames_to_render,
                                  interval=1000 // fps,
                                  repeat=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        ani.save(output_path, writer="pillow", fps=fps)
    else:
        ani.save(output_path, writer="ffmpeg", fps=fps,
                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the triangulated 3D skeleton + cameras after calibration."
    )
    parser.add_argument("--prefix",  required=True,
                        help="Example: ./data/A001_P001_G001")
    parser.add_argument("--subset",  default="noise_1_0",
                        help="Name of the subset folder with poses (default: noise_1_0)")
    parser.add_argument("--calib",   default="linear_1_0_ba",
                        help="Name of the calibration JSON file in results/ (default: linear_1_0_ba)")
    parser.add_argument("--dataset", default="MyDataset")
    parser.add_argument("--output",  default=None,
                        help="Output path .gif or .mp4 (default: results/camera/visu_3d.gif)")
    parser.add_argument("--fps",     type=int, default=15,
                        help="Animation FPS (default: 15)")
    parser.add_argument("--step",    type=int, default=0,
                        help="Render 1 frame every N (default: 0 = auto-cap GIF to ~150 frames)")
    parser.add_argument("--max_frames", type=int, default=150,
                        help="Max GIF frames when --step=0 (default: 150)")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                         help="Confidence threshold for 2D keypoints")
    parser.add_argument("--export_trc", default=None,
                         help="Export the triangulated 3D skeleton to a .trc file path")
    args = parser.parse_args()

    subset_dir = os.path.join(args.prefix, args.subset)
    calib_json = os.path.join(args.prefix, "results", f"{args.calib}.json")
    if args.output is None:
        out_dir = os.path.join(args.prefix, "results", "camera")
        args.output = os.path.join(out_dir, f"visu_3d_{args.calib}.gif")

    with open(os.path.join(_REPO_ROOT, "config", "config.yaml")) as f:
        config = yaml.safe_load(f)
    
    base_name = os.path.basename(os.path.normpath(args.prefix))
    import re
    match = re.search(r'A(\d+)_P(\d+)_G(\d+)', base_name)
    if match:
        aid, pid, gid = int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        # Fallback for folder names that don't follow the Axxx_Pxxx_Gxxx convention
        print(f"WARNING: Output directory '{base_name}' does not match Axxx_Pxxx_Gxxx format. Using default IDs (AID=1, PID=1, GID=1).")
        aid, pid, gid = 1, 1, 1
    
    camera_ids = config[args.dataset]["camera_ids"]

    print(f"Dataset    : {args.dataset}")
    print(f"Subset     : {subset_dir}")
    print(f"Calibration: {calib_json}")
    print(f"Cameras    : {camera_ids}")

    CAMID, K, R_w2c, t_w2c, _dist = load_eldersim_camera(calib_json)
    R_w2c = np.array(R_w2c)
    t_w2c = np.array(t_w2c)
    K      = np.array(K)
    print(f"Loaded calibration for {len(CAMID)} cameras")

    p2d_list, s2d_list = [], []
    min_frames = 999999
    for cid in camera_ids:
        fpath = os.path.join(subset_dir, "2d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json")
        frames, p2d, s2d = load_poses(fpath)
        if len(frames) < min_frames:
            min_frames = len(frames)
        p2d_list.append(p2d.reshape(len(frames), -1, 2))
        s2d_list.append(s2d)

    p2d_list = [p[:min_frames] for p in p2d_list]
    s2d_list = [s[:min_frames] for s in s2d_list]
    p2d_all = np.array(p2d_list)
    s2d_all = np.array(s2d_list)
    N = p2d_all.shape[1]
    print(f"Frames     : {N} (truncated to shortest video)")

    print("Triangulating 3D skeleton in world frame...")
    X3d_world = triangulate_skeleton(p2d_all, s2d_all, K, R_w2c, t_w2c, args.conf_threshold)
    print(f"X3d_world shape: {X3d_world.shape}")

    if args.export_trc:
        dataset_fps = config.get(args.dataset, {}).get("frame_rate", 30.0)
        export_to_trc(X3d_world, args.export_trc, fps=dataset_fps)

    # Auto-step: cap GIF to ~max_frames for fast rendering
    step = args.step
    if step == 0:
        step = max(1, N // args.max_frames)
        if step > 1:
            print(f"Auto-step: rendering 1/{step} frames ({N // step}/{N} total)")

    make_animation(X3d_world, R_w2c, t_w2c,
                   output_path=args.output,
                   fps=args.fps,
                   step=step)


if __name__ == "__main__":
    main()
