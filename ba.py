#%%
import cv2
from matplotlib.style import available
import numpy as np
import sys
import json
import os
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from numba import jit
import util
from argument import parse_args
from pycalib.calib import *
import yaml
from util import project_cv2


def to_theta(R, t, x):
    rvecs = np.array([cv2.Rodrigues(r)[0].flatten() for r in R]).flatten()
    return np.hstack([rvecs, t.flatten(), x.flatten()])


def from_theta(theta, C):
    # kvecs = theta[0:3*Nc]
    rvecs = theta[0 : 3 * C].reshape((C, 3))
    tvecs = theta[3 * C : 6 * C].reshape((C, 3, 1))
    x = theta[6 * C :].reshape((-1, 3))

    R = np.array([cv2.Rodrigues(r)[0] for r in rvecs])

    return R, tvecs, x


def objfun_nll(K_all, R_all, t_all, x_all, y_all, y_mask_all, s_all):
    # K_all : C x 3 x 3
    # R_all : C x 3 x 3, w2c
    # t_all : C x 3
    # x_all : N x 3
    # y_all : C x Nc
    # y_mask_all : C x N (bool index)
    # sd_all : C x Nc

    e_all = []
    for K, R, t, mask, y, s in zip(K_all, R_all, t_all, y_mask_all, y_all, s_all):
        # for each camera

        # visible points
        x = x_all[mask]
        y = y[mask]
        s = np.sqrt(2) * s[mask]
        s = np.copy(s.reshape((-1, 1)))

        # project
        t = np.copy(t)  # for jit
        y_hat = K @ (R @ x.T + t.reshape((3, 1)))
        y_hat = (y_hat[:2, :] / y_hat[2, :]).T
        e = y - y_hat[:, :2]
        e = e * s
        # e = np.sum(e**2, axis=1) / (2 * (sd**2))
        # e = np.exp(-e) #/ (2*np.pi*sd) # we do not need this to make the min of NLL be zero
        e_all.append(e.flatten())

    return np.concatenate(e_all)
    # return np.array(e_all).flatten() #np.log(np.hstack(e_all))


def objfun_var3d(R_all, vc_all, vc_mask_all, bone_idx):
    # R_all : C x 3 x 3, w2c
    # vc_all : C x N x J x 3
    # vc_mask : C x N x J, bool
    C = len(R_all)

    vw = []
    for R, vc in zip(R_all, vc_all):
        vw.append(vc @ R)
    vw = np.array(vw)
    vw[~vc_mask_all] = np.nan

    # CxNxBx2x3
    bones = vw[:, :, bone_idx, :]

    # CxNxBx3
    dirs = bones[:, :, :, 0, :] - bones[:, :, :, 1, :]

    # Cx(NB)x3
    dirs = dirs.reshape(C, -1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=2)[:, :, None]
    # print(dirs.shape)

    # points with all mask==False
    m_invalid = np.isnan(dirs).any(axis=(0, 2))
    # var[m_invalid] = 0

    var = 1 - np.linalg.norm(np.nanmean(dirs[:, ~m_invalid], axis=0), axis=1)

    return var


def objfun_varbone(x_all, bone_idx, invalid_mask=None):
    # x_all : N x J x 3
    # bone_idx : B x 2

    if invalid_mask is not None:
        x_all_copy = np.copy(x_all).reshape(-1, 3)
        x_all_copy[invalid_mask] = np.nan
        x_all = x_all_copy.reshape(x_all.shape)

    # N x B x 2 x 3
    bone = x_all[:, bone_idx, :]
    # N x B
    bone_length = np.linalg.norm(bone[:, :, 0, :] - bone[:, :, 1, :], axis=2)
    # variance
    with np.errstate(invalid='ignore'):
        bone_var = np.nanvar(bone_length, axis=0)
    bone_var[np.isnan(bone_var)] = 0.0

    return bone_var


def objfun(params, K, sp2d, ss2d, sp3d, ss3d, bone_idx, C, N, J, lambda1, lambda2, invalid_mask):
    R_w2c, t_w2c, x = from_theta(params, C)
    E = []
    e = objfun_nll(
        K,
        R_w2c,
        t_w2c,
        x,
        sp2d,
        (ss2d > 0.5).reshape((C, N * J)),
        ss2d.reshape((C, N * J)),
    )
    # print(f'Initial mean NLL: {np.mean(e*e)}')
    E.append(e.flatten())

    e = objfun_var3d(R_w2c, sp3d, (ss3d > 0), bone_idx)
    # print(f'Initial mean 3D variance: {np.mean(e*e)}')
    E.append(e * lambda1)

    e = objfun_varbone(x.reshape(N, J, 3), bone_idx, invalid_mask)
    # print(f'Initial mean bone-length variance: {np.mean(e*e)}')
    E.append(e * lambda2)

    return np.concatenate(E)


def gen_new_mask(x_all, C, J, N):

    assert x_all.shape == (N, J, 3)
    # nan -> 1,0,0 for optimization

    x_all = x_all.reshape(-1, 3)
    mask_nan = np.isnan(x_all)[:, 0]
    x_all[mask_nan] = np.array([1, 0, 0])

    mask = np.tile(~mask_nan, (C, 1))  # for more then2 views
    mask = mask.reshape(C, N, J)
    return mask, x_all


def ba_main(camid, K, R_w2c, t_w2c, sp2d, ss2d, sp3d, ss3d, lambda1, lambda2):

    C = len(camid)
    N = sp2d.shape[1]
    J = sp2d.shape[2]

    x_all = util.triangulate_with_conf(sp2d, ss2d, K, R_w2c, t_w2c, (ss2d > 0.5))
    x_all = x_all.reshape(N * J, 3)

    assert x_all.shape == (N * J, 3)

    # Remplacer les NaNs par 0.0 pour empêcher SciPy (least_squares) de planter (ValueError: `x0` is infeasible)
    invalid_mask = np.isnan(x_all).any(axis=1)
    x_all[invalid_mask] = 0.0

    # Forcer la confiance de ces points invalides à 0 pour que l'optimiseur les ignore strictement
    invalid_mask_3d = invalid_mask.reshape(N, J)
    for c in range(C):
        ss2d[c][invalid_mask_3d] = 0.0

    e = objfun_nll(
        K,
        R_w2c,
        t_w2c,
        x_all,
        sp2d.reshape((C, N * J, 2)),
        (ss2d > 0.5).reshape((C, N * J)),
        ss2d.reshape((C, N * J)),
    )
    print(f"Initial mean NLL: {np.mean(e*e)}")

    e = objfun_var3d(R_w2c, sp3d, (ss3d > 0), util.OP_BONE)
    print(f"Initial mean 3D variance: {np.mean(e * e)}")

    e = objfun_varbone(x_all.reshape(N, J, 3), util.OP_BONE, invalid_mask)
    print(f"Initial mean bone-length variance: {np.mean(e * e)}")

    theta0 = to_theta(R_w2c, t_w2c, x_all)

    res = least_squares(
        objfun,
        theta0,
        verbose=True,
        ftol=1e-4,
        method="trf",
        args=(
            K,
            sp2d.reshape((C, N * J, 2)),
            ss2d,
            sp3d,
            ss3d,
            util.OP_BONE,
            C,
            N,
            J,
            lambda1,
            lambda2,
            invalid_mask,
        ),
    )

    R_w2c_opt, t_w2c_opt, x_opt = from_theta(res["x"], C)

    return R_w2c_opt, t_w2c_opt, x_opt


def save_json(out_dir, x2d, s2d, frames, aid, pid, gid, cid, joint2d_dir):

    os.makedirs(os.path.join(out_dir, joint2d_dir), exist_ok=True)

    with open(
        os.path.join(
            out_dir, joint2d_dir, f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"
        ),
        "w",
    ) as fp:
        data = []
        for f, x, s in zip(frames, x2d, s2d):

            data.append(
                {
                    "frame_index": int(f),
                    "skeleton": [{"pose": x.flatten().tolist(), "score": s.tolist()}],
                }
            )
        json.dump({"data": data}, fp, indent=2, ensure_ascii=True)


def save_mask(intrinsic, R, t, obs_mask, width, height):

    # if bObsMask:
    print("save obs. mask")
    sCAMID_all, _, _, _, _, _, _, sp2d_all, ss2d_all, sframes_all = util.load_eldersim(
        PREFIX, GID, AID, PID
    )
    x_all = util.triangulate_with_conf(
        sp2d_all, ss2d_all, intrinsic, R, t, (ss2d_all > 0.5)
    )
    projected_x2d = project_cv2(R, t, intrinsic, x_all, width, height)

    new_mask = np.linalg.norm(sp2d_all - projected_x2d, axis=3) > TH_MASK
    ss2d_all[new_mask] = np.nan

    if obs_mask:
        joint2d_dir = "2d_joint_mask_ba"
    else:
        joint2d_dir = "2d_joint_mask"

    for i in range(len(sCAMID_all)):
        save_json(
            PREFIX,
            sp2d_all[i],
            ss2d_all[i],
            sframes_all,
            AID,
            PID,
            GID,
            sCAMID_all[i],
            joint2d_dir,
        )


if __name__ == "__main__":

    args = parse_args()
    PREFIX = (
        args.prefix
        + "/"
        + "noise_"
        + args.target.split("_")[1]
        + "_"
        + args.target.split("_")[2]
    )
    AID = args.aid
    PID = args.pid
    GID = args.gid
    OBS_MASK = args.obs_mask
    SAVE_OBS_MASK = args.save_obs_mask

    if OBS_MASK:
        JSON_IN = args.prefix + "/results/" + args.target + "_mask.json"
        JSON_OUT = args.prefix + "/results/" + args.target + "_mask_ba.json"
    else:
        JSON_IN = args.prefix + "/results/" + args.target + ".json"
        JSON_OUT = args.prefix + "/results/" + args.target + "_ba.json"
    DATASET = args.dataset
    # bObsMask = args.obs_mask
    TH_MASK = args.th_obs_mask

    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())

    width = config[DATASET]["width"]
    height = config[DATASET]["height"]
    available_joints = config[DATASET]["available_joints"]
    FRAME_SKIP = args.frame_skip

    # if OBS_MASK:
    # sCAMID, _ , _, _, sp3d_w, sp3d, ss3d, sp2d, ss2d, sframes = util.load_eldersim(PREFIX, GID, AID, PID,joint2d_dir='2d_joint_mask')
    #     print("12323132")
    # else :
    sCAMID, _, _, _, sp3d_w, sp3d, ss3d, sp2d, ss2d, sframes = util.load_eldersim(
        PREFIX, GID, AID, PID
    )

    CAMID, intrinsic, R_w2c, t_w2c = util.load_eldersim_camera(JSON_IN)
    LAMBDA1 = args.ba_lambda1
    LAMBDA2 = args.ba_lambda2
    assert np.alltrue(CAMID == sCAMID)

    sp3d_w = sp3d_w[::FRAME_SKIP, :, :]
    sp3d = sp3d[:, ::FRAME_SKIP, :, :]
    ss3d = ss3d[:, ::FRAME_SKIP, :]
    sp2d = sp2d[:, ::FRAME_SKIP, available_joints, :]
    ss2d = ss2d[:, ::FRAME_SKIP, available_joints]
    sframes = sframes[::FRAME_SKIP]

    print(f"dataset={DATASET}")
    print(f"target BA={JSON_IN}")

    R_w2c_opt, t_w2c_opt, x_opt = ba_main(
        CAMID, intrinsic, R_w2c, t_w2c, sp2d, ss2d, sp3d, ss3d, LAMBDA1, LAMBDA2
    )
    if SAVE_OBS_MASK:
        save_mask(intrinsic, R_w2c_opt, t_w2c_opt, OBS_MASK, width, height)

    with open(JSON_OUT, "w") as fp:
        out = {
            "CAMID": CAMID.tolist(),
            "K": intrinsic.tolist(),
            "R_w2c": R_w2c_opt.tolist(),
            "t_w2c": t_w2c_opt.tolist(),
        }
        json.dump(out, fp, indent=2, ensure_ascii=True)
    print(" ")
# %%
