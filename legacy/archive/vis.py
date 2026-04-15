from distutils.sysconfig import PREFIX
from os import makedirs
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # fix WSL: no display
import matplotlib
matplotlib.use("Agg")                          # fix WSL: no display
from re import L
from util import load_poses
import numpy as np
import cv2
from argument import parse_args
import matplotlib.pylab as plt
from pycalib.plot import plotCamera, axisEqual3D
from util import load_eldersim_camera, load_eldersim, triangulate_with_conf

import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d

EPOCH_1ST = 100
EPOCH_2ND = 120
OPENPOSE_SKELETON = (
    (1, 8),
    (1, 2),
    (1, 5),
    (0, 15),
    (0, 16),
    (15, 17),
    (16, 18),
    (1, 0),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (8, 12),
    (9, 10),
    (12, 13),
    (10, 11),
    (13, 14),
    (11, 24),
    (11, 22),
    (22, 23),
    (14, 21),
    (14, 19),
    (19, 20),
)


def vis_2d(prefix, target, width, height, frame_rate, aid, pid, gid, camera_ids):

    joint2d_dir = prefix + "/" + target + "/2d_joint"
    for cam_id in camera_ids:

        fjson = f"{joint2d_dir}/A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.json"
        fvideo = f"{prefix}/videos/A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.mp4"
        out_dir = f"{prefix}/results/2d_joint"
        os.makedirs(out_dir, exist_ok=True)
        fvideo_out = (
            f"{out_dir}/{target}_A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.mp4"
        )

        frame_index, p2d, s2d = load_poses(fjson)
        size = (width, height)
        p2d = p2d.reshape(p2d.shape[0], -1, 2)
        fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(fvideo_out, fmt, frame_rate, size)
        if os.path.isfile(fvideo):
            capture = cv2.VideoCapture(fvideo)
        for (j, p2d_i, s2d_i) in zip(range(frame_index.shape[0]), p2d, s2d):
            if os.path.isfile(fvideo):
                _, frame = capture.read()
            else:
                frame = np.full((height, width, 3), 255, dtype=np.uint8)
            for pos_xy, score_xy in zip(p2d_i, s2d_i):
                if (score_xy > 0).all():
                    cv2.circle(
                        frame,
                        (int(pos_xy[0]), int(pos_xy[1])),
                        3,
                        (255, 0, 0),
                        thickness=3,
                    )
            for j0, j1 in OPENPOSE_SKELETON:
                x0 = p2d_i[j0]
                x1 = p2d_i[j1]
                if not np.isnan(x0).any() and not np.isnan(x1).any():
                    cv2.line(
                        frame,
                        pt1=(int(x0[0]), int(x0[1])),
                        pt2=(int(x1[0]), int(x1[1])),
                        color=(0, 255, 0),
                        thickness=3,
                        lineType=cv2.LINE_4,
                        shift=0,
                    )
            writer.write(frame)
        writer.release()
        if os.path.isfile(fvideo):
            capture.release()
        print(f"out:{fvideo_out}")


def vis_3d_retrain(prefix, target, aid, pid, gid, camera_ids):
    joint3d_dir = prefix + "/" + target + "/3d_joint"
    for cam_id in camera_ids:

        fjson = f"{joint3d_dir}/A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.json"

        out_dir = f"{prefix}/results/3d_joint"
        os.makedirs(out_dir, exist_ok=True)
        _, p3d, _ = load_poses(fjson)
        p3d = p3d.reshape(p3d.shape[0], -1, 3)

        fgif_out = (
            f"{out_dir}/{target}_A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.gif"
        )

        draw_3d(p3d, fgif_out)


def vis_3d(prefix, target, aid, pid, gid, camera_ids, vis_type):
    joint3d_dir = prefix + "/" + target + "/3d_joint"
    for cam_id in camera_ids:

        fjson = f"{joint3d_dir}/A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.json"
        out_dir = f"{prefix}/results/3d_joint"
        _, p3d, _ = load_poses(fjson)
        p3d = p3d.reshape(p3d.shape[0], -1, 3)
        os.makedirs(out_dir, exist_ok=True)
        fgif_out = (
            f"{out_dir}/{target}_A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.gif"
        )

        if vis_type == "retrain1" or vis_type == "retrain2":
            fjson_retrain1 = f"{prefix}/noise_{aid}_{EPOCH_1ST}/3d_joint/A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.json"
            _, p3d_retrain1, _ = load_poses(fjson_retrain1)
            p3d_retrain1 = p3d_retrain1.reshape(p3d_retrain1.shape[0], -1, 3)

        if vis_type == "retrain2":
            fjson_retrain2 = f"{prefix}/noise_{aid}_{EPOCH_2ND}/3d_joint/A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cam_id:03d}.json"
            _, p3d_retrain2, _ = load_poses(fjson_retrain2)
            p3d_retrain2 = p3d_retrain2.reshape(p3d_retrain2.shape[0], -1, 3)

        if vis_type == "3d":
            draw_3d(p3d, fgif_out, vis_type=vis_type)
        elif vis_type == "retrain1":
            draw_3d(p3d, fgif_out, p3d_retrain1=p3d_retrain1, vis_type=vis_type)
        elif vis_type == "retrain2":
            draw_3d(
                p3d,
                fgif_out,
                p3d_retrain1=p3d_retrain1,
                p3d_retrain2=p3d_retrain2,
                vis_type=vis_type,
            )


def draw_3d(
    p3d,
    out,
    p3d_retrain1=None,
    p3d_retrain2=None,
    R_w2c=None,
    t_w2c=None,
    vis_scale=0.05,
    vis_type="3d",
):
    fig = plt.figure()

    if vis_type == "3d" or vis_type == "camera":
        ax1 = fig.add_subplot(projection="3d")
    elif vis_type == "retrain1":
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

    elif vis_type == "retrain2":
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133, projection="3d")

    def setLines(X, Y, Z):
        lineX = []
        lineY = []
        lineZ = []

        for bone in OPENPOSE_SKELETON:
            lineX.append([X[bone[0]], X[bone[1]]])
            lineY.append([Y[bone[0]], Y[bone[1]]])
            lineZ.append([Z[bone[0]], Z[bone[1]]])

        return np.array(lineX), np.array(lineY), np.array(lineZ)

    def draw_skeleton(ax, X, Y, Z, title):
        ax.clear()
        ax.set_title(title)
        ax.view_init(elev=-90, azim=-86)
        if R_w2c is None and t_w2c is None:
            ax.set_xlim(np.nanmin(p3d[:, :, 0]), np.nanmax(p3d[:, :, 0]))
            ax.set_ylim(np.nanmin(p3d[:, :, 1]), np.nanmax(p3d[:, :, 1]))
            ax.set_zlim(np.nanmin(p3d[:, :, 2]), np.nanmax(p3d[:, :, 2]))

        ax.plot(X, Y, Z, "k.")

        X_bone, Y_bone, Z_bone = setLines(X, Y, Z)

        for x, y, z in zip(X_bone, Y_bone, Z_bone):
            line = art3d.Line3D(x, y, z, color="#f94e3e")
            ax.add_line(line)

    def update_frame(fc):
        X = p3d[fc, :, 0]
        Y = p3d[fc, :, 1]
        Z = p3d[fc, :, 2]
        if (p3d_retrain1 is not None or p3d_retrain1 is not None):
            draw_skeleton(ax1, X, Y, Z, "w/o fine-tuning")
        else :
            draw_skeleton(ax1, X, Y, Z, "")
        if p3d_retrain1 is not None:
            X1 = p3d_retrain1[fc, :, 0]
            Y1 = p3d_retrain1[fc, :, 1]
            Z1 = p3d_retrain1[fc, :, 2]
            draw_skeleton(ax2, X1, Y1, Z1, "1st fine-tuning")
        if p3d_retrain2 is not None:
            X2 = p3d_retrain2[fc, :, 0]
            Y2 = p3d_retrain2[fc, :, 1]
            Z2 = p3d_retrain2[fc, :, 2]
            draw_skeleton(ax3, X2, Y2, Z2, "2nd fine-tuning")

        if R_w2c is not None and t_w2c is not None:
            for r, t in zip(R_w2c, t_w2c):
                plotCamera(ax1, r.T, -r.T @ t, color="b", scale=vis_scale)
            axisEqual3D(ax1)

    ani = animation.FuncAnimation(
        fig, update_frame, frames=p3d.shape[0], interval=30, repeat=False
    )
    ani.save(out, writer="pillow")
    print(f"out:{out}")


def vis_camera(prefix, target, aid, pid, gid, vis_type, subset_dir=None):

    result_dir = prefix + "/results/"
    json_in = f"{result_dir}/{target}.json"
    # Use provided subset_dir, fallback to gt_subset for compatibility
    data_dir = subset_dir if subset_dir is not None else prefix + "/gt_subset"

    CAMID, K, R_w2c, t_w2c = load_eldersim_camera(json_in)
    _, _, _, _, _, _, _, gp2d, gs2d, _ = load_eldersim(data_dir, gid, aid, pid)

    x3d_all = triangulate_with_conf(gp2d, gs2d, K, R_w2c, t_w2c, (gs2d > 0))
    out_dir = f"{prefix}/results/camera"
    os.makedirs(out_dir, exist_ok=True)

    fgif_out = f"{out_dir}/{target}_A{aid:03d}_P{pid:03d}_G{gid:03d}.gif"

    draw_3d(p3d=x3d_all, out=fgif_out, R_w2c=R_w2c, t_w2c=t_w2c, vis_type=vis_type)


if __name__ == "__main__":

    args = parse_args()

    DATASET = args.dataset
    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())

    width = config[DATASET]["width"]
    height = config[DATASET]["height"]
    available_joints = config[DATASET]["available_joints"]
    camera_ids = config[DATASET]["camera_ids"]
    frame_rate = config[DATASET]["frame_rate"]

    if args.vis_type == "2d":
        vis_2d(
            args.prefix,
            args.target,
            width,
            height,
            frame_rate,
            args.aid,
            args.pid,
            args.gid,
            camera_ids,
        )
    elif (
        args.vis_type == "3d"
        or args.vis_type == "retrain1"
        or args.vis_type == "retrain2"
    ):
        vis_3d(
            args.prefix,
            args.target,
            args.aid,
            args.pid,
            args.gid,
            camera_ids,
            args.vis_type,
        )

    elif args.vis_type == "camera":
        # Use gt_subset if available, otherwise fall back to the input subset (noise_*_*)
        gt_dir = os.path.join(args.prefix, "gt_subset")
        subset_parts = args.target.split("_")  # e.g. linear_1_0_ba -> noise_1_0
        noise_dir = os.path.join(args.prefix,
                                 f"noise_{subset_parts[1]}_{subset_parts[2]}")
        subset_dir = gt_dir if os.path.isdir(gt_dir) else noise_dir
        vis_camera(
            args.prefix, args.target, args.aid, args.pid, args.gid,
            args.vis_type, subset_dir=subset_dir
        )
