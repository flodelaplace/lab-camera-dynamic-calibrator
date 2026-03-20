#%%
import cv2
import numpy as np
import sys
import os
import json
import itertools
from copy import deepcopy
from pycalib.plot import plotCamera, axisEqual3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argument import parse_args

import util

from pycalib.calib import *


def load_pose_pairs(json2d, json3d):
    f2d, p2d, s2d = util.load_poses(json2d)
    f3d, p3d, s3d = util.load_poses(json3d)

    # select valid points only
    mask = (s2d * s3d).astype(np.bool)
    p2d = p2d.reshape((-1, 25, 2))[mask, :]
    p3d = p3d.reshape((-1, 25, 3))[mask, :]

    p2d = np.array(p2d).reshape((-1, 2))
    p3d = np.array(p3d).reshape((-1, 3))

    return p2d, p3d


def load_sync(camera_indices, json2d_fmt, json3d_fmt):
    f2d_all = []
    p2d_all = []
    s2d_all = []
    f3d_all = []
    p3d_all = []
    s3d_all = []
    for c in camera_indices:
        f2d, p2d, s2d = util.load_poses(json2d_fmt % c)
        f3d, p3d, s3d = util.load_poses(json3d_fmt % c)

        f2d_all.append(f2d)
        p2d_all.append(p2d.reshape((-1, 25, 2)))
        s2d_all.append(s2d)
        f3d_all.append(f3d)
        p3d_all.append(p3d.reshape((-1, 25, 3)))
        s3d_all.append(s3d)
        assert np.array_equal(f2d, f3d)

    return (
        np.array(p2d_all),
        np.array(p3d_all),
        np.array(s2d_all),
        np.array(s3d_all),
        np.array(f2d_all[0]),
    )


def plot(R_w2c_lists, t_w2c_lists, x3D_world, out_dir, aid, pid, gid):
    assert len(R_w2c_lists) == len(t_w2c_lists)
    assert x3D_world.shape[-1] == 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter3D(x3D_world[:, 0], x3D_world[:, 1], x3D_world[:, 2], color="g")

    for r, t in zip(R_w2c_lists, t_w2c_lists):
        plotCamera(ax, r.T, -r.T @ t, "b", 10)
    axisEqual3D(ax)

    ax.view_init(elev=-84, azim=-90)
    fig.savefig(
        f"{out_dir}/vis_A{aid:03d}_P{pid:03d}_G{gid:03d}.png", bbox_inches="tight"
    )
    return fig


def main(prefix, aid, pid, gid, cids, out_dir):

    os.makedirs(os.path.join(out_dir, "2d_joint"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "3d_joint"), exist_ok=True)

    K0 = np.eye(3)
    K0[0, 0] = K0[1, 1] = 320
    K0[0, 2] = 320
    K0[1, 2] = 180
    print("K = ", K0)

    # load
    p2d, p3d, s2d, s3d, frames = load_sync(
        cids,
        f"{prefix}/Openpose/2DJ/A{aid:03d}_P{pid:03d}_G{gid:03d}_C%03d.json",
        f"{prefix}/Openpose/3DJ/A{aid:03d}_P{pid:03d}_G{gid:03d}_C%03d.json",
    )
    mask = (s2d == 1) * (s3d == 1)
    Nc = len(p3d)

    # flip X,Y,Z (eldersim)  ->  X,-Y,Z (opencv)
    p3d[:, :, :, 1] = -p3d[:, :, :, 1]

    # Correct 3D poses iteratively
    for iter in range(3):
        for i in range(Nc):
            p2 = deepcopy(p2d[i, mask[i], :].reshape((-1, 2)))
            p3 = deepcopy(p3d[i, mask[i], :].reshape((-1, 3)))

            ret, rvec, tvec = cv2.solvePnP(p3, p2, K0, np.zeros((5, 1)))
            rmat = cv2.Rodrigues(rvec)[0]

            # R should be identity
            assert np.allclose(rmat, np.eye(3), atol=1e-6), rmat

            # print(tvec)

            e = util.reprojection_error(util.to_x(K0, rvec, tvec), p3, p2).reshape(
                (-1, 2)
            )
            # print(f'ITER{iter} G{GID}-C{i+1} reproj_err = {np.mean(np.linalg.norm(e, axis=1)):e}')

            # correct
            p3 = p3 + tvec.flatten()
            e = util.reprojection_error(
                util.to_x(K0, rvec, np.zeros(3)), p3, p2
            ).reshape((-1, 2))
            # print(f'ITER{iter} G{GID}-C{i+1} reproj_err = {np.mean(np.linalg.norm(e, axis=1)):e}')

            p2d[i, mask[i], :] = p2
            p3d[i, mask[i], :] = p3

    # Check
    for i in range(Nc):
        p2 = deepcopy(p2d[i, mask[i], :].reshape((-1, 2)))
        p3 = deepcopy(p3d[i, mask[i], :].reshape((-1, 3)))

        ret, rvec, tvec = cv2.solvePnP(p3, p2, K0, np.zeros((5, 1)))
        rmat = cv2.Rodrigues(rvec)[0]

        # R should be identity
        assert np.allclose(rmat, np.eye(3), atol=1e-6), rmat
        assert np.allclose(tvec, np.zeros(3), atol=1e-6), tvec

        # print(ret)
        e = util.reprojection_error(util.to_x(K0, rvec, tvec), p3, p2).reshape((-1, 2))
        # print(f'CHECK G{GID}-C{i+1} reproj_err = {np.mean(np.linalg.norm(e, axis=1)):e}')

    # pairwise calibration
    Rt_pairs = dict()
    for a, b in itertools.combinations(range(Nc), 2):
        p3da = deepcopy(p3d[a])
        p3db = deepcopy(p3d[b])
        s3da = s3d[a]
        s3db = s3d[b]
        m = (s3da == 1) * (s3db == 1)

        # filter occluded joints by NaN
        p3da[~m, :] = np.nan
        p3db[~m, :] = np.nan
        # print(np.sum(np.isnan(p3da)))

        p3da = p3da.reshape((-1, 3))
        p3db = p3db.reshape((-1, 3))
        m = np.min(~np.isnan(p3da), axis=1)
        p3da = p3da[m, :]
        p3db = p3db[m, :]
        # print(np.sum(np.isnan(p3da)))

        R, t, _ = absolute_orientation(p3da.T, p3db.T, no_scaling=True)
        e = p3db.T - (R @ p3da.T + t[:, None])
        e = np.linalg.norm(e, axis=0)
        # print(np.mean(e))

        Rt_pairs[a, b] = np.hstack((R, t[:, None]))

    # Registration
    R, t, err_r, err_t = pose_registration(Nc, Rt_pairs)

    # Transform to make Camera0 be WCS
    R_est = []
    t_est = []

    for c in reversed(range(Nc)):
        Rx, tx = rebase(
            R[:3, :3], t[:3], R[3 * c : 3 * c + 3, :3], t[3 * c : 3 * c + 3]
        )
        R_est.append(Rx)
        t_est.append(tx)
    R_est = np.array(R_est[::-1])
    t_est = np.array(t_est[::-1])

    # This estimation is up-to-scale.  So normalize by the cam1-cam2 distance.
    for c in reversed(range(Nc)):
        t_est[c] /= np.linalg.norm(t_est[1])

    # make sure the points are in front of the cameras
    def z_test(R_est, t_est, p2d, K, mask):
        m = mask[0] * mask[1]

        pa = p2d[0, m, :].reshape((-1, 2))
        pb = p2d[1, m, :].reshape((-1, 2))
        na = np.ones((pa.shape[0], 3), np.float64)
        nb = np.ones((pa.shape[0], 3), np.float64)
        na[:, :2] = pa
        nb[:, :2] = pb
        Ki = np.linalg.inv(K)
        na = np.copy(na @ Ki.T)
        nb = np.copy(nb @ Ki.T)
        return util.z_test_w2c(
            R_est[0], t_est[0].flatten(), R_est[1], t_est[1].flatten(), na, nb
        )

    sign, Np, Nn = z_test(R_est, t_est, p2d, K0, mask)
    print(sign, Np, Nn)
    t_est = sign * t_est

    ######### verification #########

    # Projection matrix
    P_est = []
    for i in range(Nc):
        P_est.append(K0 @ np.hstack((R_est[i], t_est[i])))
    P_est = np.array(P_est)

    # verify CCS -> 2D
    for i in range(Nc):
        x = deepcopy(p2d[i, mask[i], :].reshape((-1, 2)))
        X = deepcopy(p3d[i, mask[i], :].reshape((-1, 3)))

        y, _ = cv2.projectPoints(
            X, cv2.Rodrigues(np.eye(3))[0], np.zeros(3), K0, np.zeros(5)
        )
        y = y.reshape((-1, 2))
        e = np.mean(np.linalg.norm(x - y, axis=1))

        y = K0 @ X.T
        y = y[:2, :] / y[2, :]
        e = np.mean(np.linalg.norm(x - y[:2, :].T, axis=1))
        # print(f'G{GID}-C{i+1} reproj_err = {e:e}')

    # verify 2D from all camera -> triangulate -> 2D
    X = []
    E = []

    for i in range(p2d.shape[1]):
        for j in range(p2d.shape[2]):
            x = p2d[:, i, j, :].reshape((Nc, 2))
            m = mask[:, i, j]

            x3d = triangulate(x[m], P_est[m])

            X.append(x3d[:3])

            y = P_est[m] @ x3d
            y = y[:, :2] / y[:, 2:3]

            e = np.mean(np.linalg.norm(x[m] - y, axis=1))
            E.append(e)
    print(
        f"G{gid} Reprojection error: min / mean / max = {np.min(E):e} / {np.mean(E):e} / {np.max(E):e}"
    )
    X = np.array(X).reshape((p2d.shape[1], p2d.shape[2], 3))

    # verify 3D -> 3D
    E = []
    for i in range(Nc):
        # 3D points reconstructed up to scale
        x = R_est[i] @ X[mask[i]].reshape((-1, 3)).T + t_est[i]

        # Original 3D
        y = deepcopy(p3d[i][mask[i]]).reshape((-1, 3)).T

        rmat, tvec, scale = absolute_orientation(x, y, no_scaling=False)
        # R should be identity
        assert np.allclose(rmat, np.eye(3), atol=1e-4), rmat
        assert np.allclose(tvec, np.zeros(3), atol=1e-3), tvec
        e = np.linalg.norm(scale * x - y, axis=0)
        E.append(np.mean(e))
    print(
        f"G{gid} Registration error: min / mean / max = {np.min(E):e} / {np.mean(E):e} / {np.max(E):e}"
    )

    ######### Rotate the cameras as close as possible to the original ##########

    # scale
    t_est = t_est * scale
    X = X * scale

    return p2d, frames, R_est, t_est, s2d, s3d, K0, X


if __name__ == "__main__":

    args = parse_args()
    PREFIX = args.prefix
    AID = args.aid
    PID = args.pid
    GID = args.gid
    TARGET = args.target
    TARGET_CAMERAS = args.calib
    ALL_CAMERAS = list(range(1, 29))

    if len(TARGET_CAMERAS) < 2:
        TARGET_CAMERAS = ALL_CAMERAS

    print(f"calibration target = {TARGET_CAMERAS}")

    OUT_DIR = f"{args.target}/A{args.aid:03d}_P{args.pid:03d}_G{args.gid:03d}/gt_subset"

    p2d, frames, R_est, t_est, s2d, s3d, intrinsic, x3d = main(
        PREFIX, AID, PID, GID, TARGET_CAMERAS, OUT_DIR
    )

    ########## save_subset #########

    # p2d_subset = p2d[TARGET_CAMERAS]
    R_est_subset = R_est
    t_est_subset = t_est
    s2d_subset = s2d
    s3d_subset = s3d
    fig = plot(
        R_est_subset,
        t_est_subset,
        x3d[:, util.OP_KEY["MidHip"], :],
        OUT_DIR,
        args.aid,
        args.pid,
        args.gid,
    )

    with open(os.path.join(OUT_DIR, f"cameras_G{GID:03d}.json"), "w") as fp:
        out = {
            "CAMID": [i for i in TARGET_CAMERAS],
            "K": [intrinsic.tolist() for k in range(1, len(TARGET_CAMERAS) + 1)],
            "R_w2c": R_est_subset.tolist(),
            "t_w2c": t_est_subset.tolist(),
        }
        json.dump(out, fp, indent=2, ensure_ascii=True)

    with open(os.path.join(OUT_DIR, f"skeleton_w_G{GID:03d}.json"), "w") as fp:
        out = {"skeleton": x3d.tolist(), "frame_indices": frames.tolist()}
        json.dump(out, fp, indent=2, ensure_ascii=True)

    for i, cid in enumerate(TARGET_CAMERAS):
        with open(
            os.path.join(
                OUT_DIR, "3d_joint", f"A{AID:03d}_P{PID:03d}_G{GID:03d}_C{cid:03d}.json"
            ),
            "w",
        ) as fp:
            data = []
            for f, s, x in zip(frames, s3d_subset[i], x3d):
                p = R_est_subset[i] @ x.T + t_est_subset[i]
                data.append(
                    {
                        "frame_index": int(f),
                        "skeleton": [
                            {"pose": p.T.flatten().tolist(), "score": s.tolist()}
                        ],
                    }
                )
            json.dump({"data": data}, fp, indent=2, ensure_ascii=True)

        with open(
            os.path.join(
                OUT_DIR, "2d_joint", f"A{AID:03d}_P{PID:03d}_G{GID:03d}_C{cid:03d}.json"
            ),
            "w",
        ) as fp:
            data = []
            for f, s, x in zip(frames, s2d_subset[i], x3d):
                p = R_est_subset[i] @ x.T + t_est_subset[i]
                p = intrinsic @ p
                p[:2, :] /= p[2, :]
                data.append(
                    {
                        "frame_index": int(f),
                        "skeleton": [
                            {"pose": p[:2, :].T.flatten().tolist(), "score": s.tolist()}
                        ],
                    }
                )
            json.dump({"data": data}, fp, indent=2, ensure_ascii=True)
