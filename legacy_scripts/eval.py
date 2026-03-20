#%%
import cv2
import os
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.calib import absolute_orientation
import util
from argument import parse_args
from pycalib.calib import *
from pycalib.plot import plotCamera
import yaml, json


def plot(R_w2c_lists, t_w2c_lists):
    assert len(R_w2c_lists) == len(t_w2c_lists)

    fig = plt.figure()
    ax = Axes3D(fig)
    for R_list, t_list in zip(R_w2c_lists, t_w2c_lists):
        for r, t in zip(R_list, t_list):
            plotCamera(ax, r.T, -r.T @ t, "b", 10)

    return fig


def align(gR_w2c, gt_w2c, R_w2c, t_w2c, X3D):

    assert X3D.shape[1] == 3
    assert np.all(gR_w2c.shape == R_w2c.shape)
    assert np.all(gt_w2c.shape == t_w2c.shape)
    R1 = gR_w2c
    R2 = R_w2c
    t1 = gt_w2c
    t2 = t_w2c
    p1 = []
    p2 = []
    for i in range(len(R_w2c)):
        p1.append(-R1[i].T @ t1[i])
        p2.append(-R2[i].T @ t2[i])
    p1 = np.array(p1).reshape((-1, 3)).T
    p2 = np.array(p2).reshape((-1, 3)).T

    R, t, s = absolute_orientation(p2, p1)

    x3d_new = s * R @ X3D.T + t[:, None]

    R2n = []
    t2n = []
    Rc2w = []
    for i in range(len(R1)):
        R2n.append(R2[i] @ R.T)
        Rc2w.append((R @ R2[i].T))
        x = -R2[i].T @ t2[i]
        x = s * R @ x + t
        t2n.append((s * R @ (-R2[i].T @ t2[i]) + t.reshape(3, 1)))
    R2n = np.array(R2n)
    t2n = np.array(t2n)
    Rc2w = np.array(Rc2w)
    Rw2c, tw2c = util.invRT_batch(Rc2w, t2n)

    return Rw2c, tw2c[:, :, None], R2n, t2n, x3d_new.T


def create_val_summary(val):
    return {
        "min": f"{np.min(val):.04f}",
        "median": f"{np.median(val):.04f}",
        "mean": f"{np.mean(val):.04f}",
        "max": f"{np.max(val):.04f}",
    }


def calc_repro_error(
    mask_2ds, frame_skip, x_all, gp2d, gK, height, width, R_w2c, t_w2c
):

    C, F, J = mask_2ds.shape
    x_est_all = np.full((C, F, J, 2), np.nan)
    x_gt_all = np.full((C, F, J, 2), np.nan)
    x_est_homo = []
    Ep_list = []
    for i in range(C):
        mask_2d = mask_2ds[i, ::frame_skip]
        x_gt = gp2d[i, ::frame_skip][mask_2d]
        np3d_w_i = x_all[::frame_skip][mask_2d]

        np3d_w_i = np3d_w_i.reshape(-1, 3)

        P_est = gK[i] @ np.hstack((R_w2c[i], t_w2c[i]))
        x_est = P_est @ np.hstack([np3d_w_i, np.ones((np3d_w_i.shape[0], 1))]).T
        x_est = x_est.T

        x_est_homo.append(x_est)

        x_est = x_est[:, :2] / x_est[:, 2:3]
        x_est_all[i, ::frame_skip][mask_2d] = x_est
        x_gt_all[i, ::frame_skip][mask_2d] = x_gt
        e = np.linalg.norm(x_est - x_gt, axis=1)

        Ep_list.append(e)

    return np.concatenate(Ep_list), x_gt_all, x_est_all, x_est_homo


def eval_main(
    gK,
    R_w2c,
    t_w2c,
    gR_w2c,
    gt_w2c,
    gp2d,
    gs2d,
    gp3d_w,
    frame_skip,
    width,
    height,
    json_out,
    scale,
):
    E_R = []
    E_t = []

    C = gp2d.shape[0]
    N = gp2d.shape[1]
    J = gp2d.shape[2]
    mask_2ds = gs2d > 0

    x_all = util.triangulate_with_conf(gp2d, gs2d, gK, R_w2c, t_w2c, (gs2d > 0))

    x_all = x_all.reshape(-1, 3)
    nR_w2c, nt_w2c, _, _, np3d_w = align(gR_w2c, gt_w2c, R_w2c, t_w2c, x_all)
    x_all = x_all.reshape(N, J, 3)

    mask_3dw = np.all(~np.isnan(gp3d_w), axis=2)
    np3d_w = np3d_w.reshape(N, J, 3)

    E_x = np.linalg.norm(np3d_w[mask_3dw] - gp3d_w[mask_3dw], axis=1) * scale
    for i in range(C):
        E_R.append(util.eval_R(gR_w2c[i], nR_w2c[i]))
        E_t.append(util.eval_t(gt_w2c[i], nt_w2c[i]))
    E_R = np.array(E_R)
    E_t = np.array(E_t) * scale
    E_p, _, _, _ = calc_repro_error(
        mask_2ds, frame_skip, x_all, gp2d, gK, height, width, R_w2c, t_w2c
    )

    print(f"       Min  Median  Mean  Max")
    print(
        f"E_R: {np.min(E_R):.04f} {np.median(E_R):.04f} {np.mean(E_R):.04f} {np.max(E_R):.04f}"
    )
    print(
        f"E_t: {np.min(E_t):.04f} {np.median(E_t):.04f} {np.mean(E_t):.04f} {np.max(E_t):.04f}"
    )
    print(
        f"E_p: {np.min(E_p):.04f} {np.median(E_p):.04f} {np.mean(E_p):.04f} {np.max(E_p):.04f}"
    )
    print(
        f"E_x: {np.min(E_x):.04f} {np.median(E_x):.04f} {np.mean(E_x):.04f} {np.max(E_x):.04f}"
    )

    with open(json_out, "w") as f:
        json.dump(
            {
                "E_R": create_val_summary(E_R),
                "E_t": create_val_summary(E_t),
                "E_p": create_val_summary(E_p),
                "E_x": create_val_summary(E_x),
            },
            f,
        )


if __name__ == "__main__":

    args = parse_args()
    GT_DIR = args.prefix + "/gt_subset"
    AID = args.aid
    PID = args.pid
    GID = args.gid
    FRAME_SKIP = args.frame_skip
    JSON_IN = args.prefix + "/results/" + args.target + ".json"
    os.makedirs(args.prefix + "/results/eval/", exist_ok=True)
    JSON_OUT = args.prefix + "/results/eval/" + args.target + ".json"
    DATASET = args.dataset

    print(f"target={JSON_IN}")
    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())

    width = config[DATASET]["width"]
    height = config[DATASET]["height"]
    scale = config[DATASET]["scale"]
    (
        gCAMID,
        gK,
        gR_w2c,
        gt_w2c,
        gp3d_w,
        gp3d,
        gs3d,
        gp2d,
        gs2d,
        gframes,
    ) = util.load_eldersim(GT_DIR, GID, AID, PID)

    CAMID, K, R_w2c, t_w2c = util.load_eldersim_camera(JSON_IN)

    eval_main(
        gK,
        R_w2c,
        t_w2c,
        gR_w2c,
        gt_w2c,
        gp2d,
        gs2d,
        gp3d_w,
        FRAME_SKIP,
        width,
        height,
        JSON_OUT,
        scale,
    )

    print(" ")
