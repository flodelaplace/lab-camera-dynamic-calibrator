import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import json
import os
import sys
import functools
import nvgpu
import torch

module_path = os.path.abspath(os.path.join("./pycalib/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import glob

from pycalib.calib import triangulate

OP_KEY = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24,
}

COCO_KEY = {
    "Nose": 0,
    "L_Eye": 1,
    "R_Eye": 2,
    "L_Ear": 3,
    "R_Ear": 4,
    "L_Shoulder": 5,
    "R_Shoulder": 6,
    "L_Elbow": 7,
    "R_Elbow": 8,
    "L_Wrist": 9,
    "R_Wrist": 10,
    "L_Hip": 11,
    "R_Hip": 12,
    "L_Knee": 13,
    "R_Knee": 14,
    "L_Ankle": 15,
    "R_Ankle": 16,
}

H36M32_KEY = {
    "Pelvis": 0,
    "R_Hip": 1,
    "R_Knee": 2,
    "R_Ankle": 3,
    "L_Hip": 6,
    "L_Knee": 7,
    "L_Ankle": 8,
    "Spin": 12,
    "Thorax": 13,
    "Nose": 14,
    "Head": 15,
    "L_Shoulder": 17,
    "L_Elbow": 18,
    "L_Wrist": 19,
    "R_Shoulder": 25,
    "R_Elbow": 26,
    "R_Wrist": 27,
}

op_to_coco = {
    "Nose": "Nose",
    "LEye": "L_Eye",
    "REye": "R_Eye",
    "LEar": "L_Ear",
    "REar": "R_Ear",
    "LShoulder": "L_Shoulder",
    "RShoulder": "R_Shoulder",
    "LElbow": "L_Elbow",
    "RElbow": "R_Elbow",
    "LWrist": "L_Wrist",
    "RWrist": "R_Wrist",
    "LHip": "L_Hip",
    "RHip": "R_Hip",
    "LKnee": "L_Knee",
    "RKnee": "R_Knee",
    "LAnkle": "L_Ankle",
    "RAnkle": "R_Ankle",
}

H36M17_KEY = {
    "Pelvis": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Spin": 7,
    "Thorax": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}


OP_BONE = np.array(
    [
        [OP_KEY["MidHip"], OP_KEY["Neck"]],
        [OP_KEY["RShoulder"], OP_KEY["LShoulder"]],
        [OP_KEY["RShoulder"], OP_KEY["RElbow"]],
        [OP_KEY["LShoulder"], OP_KEY["LElbow"]],
        [OP_KEY["RWrist"], OP_KEY["RElbow"]],
        [OP_KEY["LWrist"], OP_KEY["LElbow"]],
        [OP_KEY["RHip"], OP_KEY["MidHip"]],
        [OP_KEY["LHip"], OP_KEY["MidHip"]],
        [OP_KEY["RHip"], OP_KEY["RKnee"]],
        [OP_KEY["RKnee"], OP_KEY["RAnkle"]],
        [OP_KEY["LHip"], OP_KEY["LKnee"]],
        [OP_KEY["LKnee"], OP_KEY["LAnkle"]],
    ],
    dtype=np.int,
)


OP_KEY_SUB = np.sort(np.unique(OP_BONE.flatten()))


def mk_bone_sub(bones, subjoints):
    k = subjoints
    v = np.arange(len(subjoints))
    sidx = k.argsort()
    return v[sidx[np.searchsorted(k, OP_BONE, sorter=sidx)]]


OP_BONE_SUB = mk_bone_sub(OP_BONE, OP_KEY_SUB)


def z_test_w2c(R1, t1, R2, t2, n1, n2):
    def triangulate(R1, t1, R2, t2, n1, n2):
        Xh = cv2.triangulatePoints(
            np.hstack([R1, t1[:, None]]),
            np.hstack([R2, t2[:, None]]),
            n1[:, :2].T,
            n2[:, :2].T,
        )
        Xh /= Xh[3, :]
        return Xh[:3, :].T

    def z_count(R, t, Xw_Nx3):
        X = R @ Xw_Nx3.T + t.reshape((3, 1))
        return np.sum(X[2, :] > 0)

    Xp = triangulate(R1, t1, R2, t2, n1, n2)
    Xn = triangulate(R1, -t1, R2, -t2, n1, n2)
    zp = z_count(R1, t1, Xp) + z_count(R2, t2, Xp)
    zn = z_count(R1, t1, Xn) + z_count(R2, t2, Xn)
    return 1 if zp > zn else -1, zp, zn


def eval_R(R1, R2):
    def thetaR(R):
        e = (np.trace(R) - 1.0) / 2.0
        if e > 1:
            e = 1
        elif e < -1:
            e = -1
        return np.arccos(e)

    def logR(R):
        t = thetaR(R)
        if t == 0:
            return 0
        else:
            t / (2 * np.sin(t)) * (R - R.T)

    return np.linalg.norm(thetaR(R1.T @ R2)) / np.sqrt(2)


def eval_t(t1, t2):
    return np.linalg.norm(t1.flatten() - t2.flatten())


def to_x(K, rvec, tvec):
    return np.hstack([K[0, 0], K[0, 2], K[1, 2], rvec.flatten(), tvec.flatten()])


def from_x(x):
    K = np.eye(3)
    K[0, 0] = K[1, 1] = x[0]
    K[0, 2] = x[1]
    K[1, 2] = x[2]
    rvec = x[3:6]
    tvec = x[6:9]
    return K, rvec, tvec


def reprojection_error(x, p3d, p2d):
    K, rvec, tvec = from_x(x)
    q, _ = cv2.projectPoints(p3d, rvec, tvec, K, np.zeros(5))
    q = q.reshape((-1, 2))
    return (q - p2d).flatten()


def reprojection_error_all(
    K_list, R_w2c_list, t_w2c_list, p3d_Nx3, p2d_CxNx2, mask_CxN
):
    E = []
    for k, r, t, p2d, m in zip(K_list, R_w2c_list, t_w2c_list, p2d_CxNx2, mask_CxN):
        x = cv2.projectPoints(
            p3d_Nx3[m, :].T, cv2.Rodrigues(r)[0], t.flatten(), k, None
        )[0].reshape((-1, 2))
        E.append(np.array(x - p2d[m, :]))

    return E


def triangulate(pt2d, P):
    """
    Triangulate a 3D point from two or more views by DLT.
    """
    N = len(pt2d)
    assert N == len(P)

    AtA = np.zeros((4, 4))
    x = np.zeros((2, 4))
    for i in range(N):
        x[0, :] = P[i][0, :] - pt2d[i][0] * P[i][2, :]
        x[1, :] = P[i][1, :] - pt2d[i][1] * P[i][2, :]
        AtA += x.T @ x

    _, v = np.linalg.eigh(AtA)
    if np.isclose(v[3, 0], 0):
        return v[:, 0]
    else:
        return v[:, 0] / v[3, 0]


def triangulate_all(K_list, R_w2c_list, t_w2c_list, p2d_CxNx2, mask_CxN):
    Nc = len(K_list)
    Np = p2d_CxNx2.shape[1]
    assert K_list.shape == (Nc, 3, 3)
    assert R_w2c_list.shape == (Nc, 3, 3)
    assert t_w2c_list.shape == (Nc, 3, 1)
    assert p2d_CxNx2.shape == (Nc, Np, 2)
    assert mask_CxN.shape == (Nc, Np)
    assert mask_CxN.dtype == np.bool

    P_est = []
    for i in range(Nc):
        P_est.append(K_list[i] @ np.hstack((R_w2c_list[i], t_w2c_list[i])))
    P_est = np.array(P_est)

    X = []
    for i in range(Np):
        x = p2d_CxNx2[:, i, :].reshape((Nc, 2))
        m = mask_CxN[:, i]

        x3d = triangulate(x[m], P_est[m])
        X.append(x3d[:3])
    X = np.array(X)
    return X


def constraint_mat_from_single_view(p, proj_mat):
    u, v = p
    const_mat = np.empty((2, 4))
    const_mat[0, :] = u * proj_mat[2, :] - proj_mat[0, :]
    const_mat[1, :] = v * proj_mat[2, :] - proj_mat[1, :]

    return const_mat[:, :3], -const_mat[:, 3]


def constraint_mat(p_stack, proj_mat_stack):
    lhs_list = []
    rhs_list = []
    for p, proj in zip(p_stack, proj_mat_stack):
        lhs, rhs = constraint_mat_from_single_view(p, proj)
        lhs_list.append(lhs)
        rhs_list.append(rhs)
    A = np.vstack(lhs_list)
    b = np.hstack(rhs_list)
    return A, b


def triangulate_point(p_stack, proj_mat_stack, confs=None):
    A, b = constraint_mat(p_stack, proj_mat_stack)
    if confs is None:
        confs = np.ones(b.shape)
    else:
        confs = np.array(confs).repeat(2)

    p_w, _, rank, _ = np.linalg.lstsq(A * confs.reshape((-1, 1)), b * confs, rcond=None)

    if np.sum(confs > 0) <= 2:
        return np.full((3), np.nan)

    if rank < 3:
        raise Exception("not enough constraint")
    return p_w


def triangulate_with_conf(p2d, s2d, K, R_w2c, t_w2c, mask):

    assert p2d.ndim == 4
    assert s2d.ndim == 3

    Nc, Nf, Nj, _ = p2d.shape

    # Nc = len(CAMERAS)
    P_est = []
    for i in range(Nc):
        P_est.append(K[i] @ np.hstack((R_w2c[i], t_w2c[i])))
    P_est = np.array(P_est)

    X = []
    for i in range(Nf):
        for j in range(Nj):

            x = p2d[:, i, j, :].reshape((Nc, 2))
            m = mask[:, i, j]
            confi = s2d[:, i, j]

            if confi.sum() > 0 and m.sum() > 1:
                x3d = triangulate_point(x[m], P_est[m], confi[m])
            else:
                x3d = np.full(4, np.nan)
                # x3d = np.array([1,0,0,0])
            X.append(x3d[:3])
    X = np.array(X)
    X = X.reshape(Nf, Nj, 3)
    return X


def load_poses(filename):
    with open(filename, "r") as fp:
        P = json.load(fp)
    # print(filename)

    frame_index = []
    pose = []
    score = []
    for frame in P["data"]:
        frame_index.append(int(frame["frame_index"]))
        pose.append(np.array(frame["skeleton"][0]["pose"], dtype=np.float64))
        score.append(np.array(frame["skeleton"][0]["score"], dtype=np.float64))

    frame_index = np.array(frame_index)
    pose = np.array(pose)
    score = np.array(score)

    return frame_index, pose, score


def load_eldersim_camera(filename):
    with open(filename, "r") as fp:
        cameras = json.load(fp)

    CAMID = np.array(cameras["CAMID"], dtype=np.int)
    K = np.array(cameras["K"], dtype=np.float64)
    R_w2c = np.array(cameras["R_w2c"], dtype=np.float64)
    t_w2c = np.array(cameras["t_w2c"], dtype=np.float64)
    assert len(CAMID) == len(K)
    assert len(CAMID) == len(R_w2c)
    assert len(CAMID) == len(t_w2c)

    return CAMID, K, R_w2c, t_w2c


def load_eldersim_skeleton_w(filename):
    with open(filename, "r") as fp:
        skeleton = json.load(fp)
        p3d_w = np.array(skeleton["skeleton"], dtype=np.float64)
        frames = np.array(skeleton["frame_indices"], dtype=np.int)

    assert len(p3d_w) == len(frames)

    return p3d_w, frames


def load_poses_all(dirname, cameras, aid, pid, gid):
    # load 2d or 3d joints in a directory
    f_all = []
    p_all = []
    s_all = []
    for cid in cameras:
        f, p, s = load_poses(
            os.path.join(dirname, f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json")
        )
        N = len(f)
        f_all.append(f)
        p_all.append(p.reshape((N, 25, -1)))  # to work both for 2D and 3D
        s_all.append(s)

    return np.array(f_all), np.array(p_all), np.array(s_all)


def load_eldersim(dirname, gid, aid, pid, joint2d_dir="2d_joint"):
    CAMID, K, R_w2c, t_w2c = load_eldersim_camera(
        os.path.join(dirname, f"cameras_G{gid:03d}.json")
    )
    p3d_w, frames = load_eldersim_skeleton_w(
        os.path.join(dirname, f"skeleton_w_G{gid:03d}.json")
    )

    f2d_all = []
    p2d_all = []
    s2d_all = []
    f3d_all = []
    p3d_all = []
    s3d_all = []
    for cid in CAMID:
        f2d, p2d, s2d = load_poses(
            os.path.join(
                dirname,
                joint2d_dir,
                f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json",
            )
        )
        f2d_all.append(f2d)
        p2d_all.append(p2d.reshape((-1, 25, 2)))
        s2d_all.append(s2d)

        f3d, p3d, s3d = load_poses(
            os.path.join(
                dirname, "3d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"
            )
        )
        f3d_all.append(f3d)
        p3d_all.append(p3d.reshape((-1, 25, 3)))
        s3d_all.append(s3d)

        assert np.alltrue(f2d == f3d)

    # check frames
    f2d_common = functools.reduce(np.intersect1d, [frames, *f2d_all])

    # reject frames if we have less than 2 views
    # for f2d, s2d, s3d in zip(f2d_all, s2d_all, s3d_all):
    #    _, _, idx = np.intersect1d(f2d_common, f2d, return_indices=True)

    # extract only the common frames
    p2d_common = []
    s2d_common = []
    p3d_common = []
    s3d_common = []
    for f2d, p2d, s2d, p3d, s3d in zip(f2d_all, p2d_all, s2d_all, p3d_all, s3d_all):
        _, _, idx = np.intersect1d(f2d_common, f2d, return_indices=True)
        p2d_common.append(p2d[idx])
        s2d_common.append(s2d[idx])
        p3d_common.append(p3d[idx])
        s3d_common.append(s3d[idx])

    return (
        CAMID,
        K,
        R_w2c,
        t_w2c,
        p3d_w,
        np.array(p3d_common),
        np.array(s3d_common),
        np.array(p2d_common),
        np.array(s2d_common),
        f2d_common,
    )


def visible_from_all_cam(mask_CxNxJ):
    # find joints visible from all cams
    mask = np.min(mask_CxNxJ, axis=0)
    return mask


def joints2orientations(p3d_CxNxJx3, mask_CxNxJ, bones_Jx2):
    C = p3d_CxNxJx3.shape[0]
    N = p3d_CxNxJx3.shape[1]
    J = p3d_CxNxJx3.shape[2]
    B = bones_Jx2.shape[0]
    assert p3d_CxNxJx3.shape[3] == 3
    assert mask_CxNxJ.shape == (C, N, J)
    assert bones_Jx2.shape[1] == 2
    assert mask_CxNxJ.dtype == np.bool

    p3d_CxNxJx3 = np.copy(p3d_CxNxJx3)

    # fill occluded joints by NaN
    p3d_CxNxJx3[~mask_CxNxJ] = np.nan


    # endpoints of each bone
    pairs = p3d_CxNxJx3[:, :, bones_Jx2, :]
    # print('pairs = ', pairs.shape)

    # e1 - e0
    dirs = pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]
    assert dirs.shape[-1] == 3
    # print('dirs = ', dirs.shape)
    # print('dirs nan = ', np.sum(np.isnan(dirs)))

    # dirs.shape == Ndirs * 3
    dirs = dirs.reshape((C, N * B, 3))
    # print('dirs = ', dirs.shape)

    # delete dirs with NaN
    mask = np.min(~np.isnan(dirs), axis=(0, 2))
    dirs = dirs[:, mask, :]
    # idx = np.isnan(dirs).any(axis=(0,2))
    # dirs = dirs[:,~idx,:]
    # print('dirs = ', dirs.shape)
    # print('dirs nan = ', np.sum(np.isnan(dirs)))

    # normalize
    norm = np.linalg.norm(dirs, axis=2)
    # print(norm.shape)
    assert np.alltrue(norm > 0)
    dirs = dirs / norm[:, :, None]
    # print(dirs.shape)

    # verify
    assert dirs.shape[2] == 3
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1)

    return dirs


def joints2projections(p2d_CxNxJx2, mask_CxNxJ, joints_J):
    C = p2d_CxNxJx2.shape[0]
    N = p2d_CxNxJx2.shape[1]
    J = p2d_CxNxJx2.shape[2]
    assert p2d_CxNxJx2.shape[3] == 2
    assert mask_CxNxJ.shape == (C, N, J)
    assert mask_CxNxJ.dtype == np.bool

    # fill occluded joints by NaN
    p2d_CxNxJx2[~mask_CxNxJ] = np.nan

    # select target joints
    # p2d = p2d_CxNxJx2[:,:,joints_J,:]
    # p2d = p2d.reshape((C,N*len(joints_J), 2))
    p2d = p2d_CxNxJx2.reshape((C, -1, 2))
    idx = np.isnan(p2d).any(axis=(0, 2))
    p2d = p2d[:, ~idx, :]

    # verify
    assert p2d.shape[0] == C
    assert p2d.shape[2] == 2

    return p2d


def project(K, R_w2c, t_w2c, pts3d_w):
    p = K @ (R_w2c @ pts3d_w.T + t_w2c[:, None])
    p = p / p[2, :]
    return p.T
    # pts3d_w = np.copy(pts3d_w)
    # return cv2.projectPoints(pts3d_w, cv2.Rodrigues(R_w2c)[0], t_w2c, K, None)[0].reshape((-1, 2))


# def reprojection_error(K, R_w2c, t_w2c, pts3d_w, pts2d):
#    return pts2d[:,:2] - project(K, R_w2c, t_w2c, pts3d_w)[:,:2]


def invRT_batch(R_w2c_gt, t_w2c_gt):
    t_c2w_gt = []
    R_c2w_gt = []

    if len(t_w2c_gt.shape) == 2:
        t_w2c_gt = t_w2c_gt[:, :, None]

    for R_w2c_gt_i, t_w2c_gt_i in zip(R_w2c_gt, t_w2c_gt):
        R_c2w_gt_i, t_c2w_gt_i = invRT(R_w2c_gt_i, t_w2c_gt_i)
        R_c2w_gt.append(R_c2w_gt_i)
        t_c2w_gt.append(t_c2w_gt_i)

    t_c2w_gt = np.array(t_c2w_gt)
    R_c2w_gt = np.array(R_c2w_gt)

    return R_c2w_gt, t_c2w_gt


def invRT(R, t):
    T = np.eye(4)
    if t.shape == (3, 1):
        t = t[:, -1]

    T[:3, :3] = R
    T[:3, 3] = t
    invT = np.linalg.inv(T)
    invR = invT[0:3, 0:3]
    invt = invT[0:3, 3]
    return invR, invt


def select_gpu(i_selected_gpu=None):
    if i_selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{i_selected_gpu}"
        print(f"CUDA_VISIBLE_DEVICES={i_selected_gpu}")
        return

    unused_max = 0
    is_free_gpu = False
    try:
        gpu_info = nvgpu.gpu_info()
    except BaseException:
        traceback.print_exc()

    for i_gpu, gpu in reversed(list(enumerate(gpu_info))):
        unused = gpu["mem_total"] - gpu["mem_used"]
        if unused > unused_max:
            unused_i_gpu = i_gpu
            unused_max = unused
        # use this gpu
        if gpu["mem_used"] < 18:
            i_selected_gpu = i_gpu
            is_free_gpu = True
            break
    # There is no free GPU, use less used one.
    if i_selected_gpu is None:
        i_selected_gpu = unused_i_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{i_selected_gpu}"
    print(f"CUDA_VISIBLE_DEVICES={i_selected_gpu}, is_free:{is_free_gpu}")
    # using flag
    if is_free_gpu:
        torch.zeros(2 * 10**4, dtype=torch.float64).cuda()


# def project_cv2_ba(Rs, ts, Ks, X):

#     assert Rs.ndim == 3
#     assert Ks.ndim == 3
#     assert ts.ndim == 3  # (C,3,1)

#     Nc, _, _ = Rs.shape
#     Nf, Nj, _ = X.shape
#     X = X.reshape(Nf*Nj, 3)
#     x_out = []
#     for R, t, K in zip(Rs, ts, Ks):
#         rvec = cv2.Rodrigues(R)[0]
#         x, _ = cv2.projectPoints(X[None, :, :], rvec, t, K, np.zeros(0))
#         x = x[:, -1, :]
#         x_out.append(x)
#     x_out = np.array(x_out)
#     x_out = x_out.reshape(Nc, Nf, Nj, 2)
#     return np.array(x_out)


def save_cam(Rw2cs, tw2cs, Ks, dists, dst, gid, CAMERAS):
    print(f"save_camera:{dst}")
    with open(os.path.join(dst, f"cameras_G{gid:03d}.json"), "w") as fp:
        out = {
            "CAMID": [i for i in CAMERAS],
            "K": [k.tolist() for k in Ks],
            "dist": dists[:, :5].tolist(),
            "R_w2c": Rw2cs.tolist(),
            "t_w2c": tw2cs.tolist(),
        }
        json.dump(out, fp, indent=2, ensure_ascii=True)


def project_cv2_nan(Rs, ts, Ks, X, width, height):

    assert Rs.ndim == 3
    assert Ks.ndim == 3
    assert ts.ndim == 3  # (C,3,1)

    Nc, _, _ = Rs.shape
    Nf, Nj, _ = X.shape
    X = X.reshape(Nf * Nj, 3)
    x_out = []
    for R, t, K in zip(Rs, ts, Ks):
        rvec = cv2.Rodrigues(R)[0]
        x, _ = cv2.projectPoints(X[None, :, :], rvec, t, K, np.zeros(0))
        x = x[:, -1, :]
        x[np.any(x < 0, axis=1), :] = np.nan
        x[x[:, 0] > width, :] = np.nan
        x[x[:, 1] > height, :] = np.nan
        x_out.append(x)
    x_out = np.array(x_out)
    x_out = x_out.reshape(Nc, Nf, Nj, 2)
    return np.array(x_out)


def get_ccs(Rs_w2c, ts_w2c, X3d_w_coco, vis_mask_2d_3d):

    assert X3d_w_coco.ndim == 3
    Nf, Nj, _ = X3d_w_coco.shape

    X3d_w_coco = X3d_w_coco.reshape(Nf * Nj, 3)
    # Rs_w2c, ts_w2c = invRT_batch(Rs_c2w, ts_c2w)
    # ts_w2c = ts_w2c[:, :, None]
    X3d_c_coco = []

    for R_w2c, t_w2c, vis_mask_2d_3d_i in zip(Rs_w2c, ts_w2c, vis_mask_2d_3d):
        Xc_coco = (R_w2c @ X3d_w_coco.T + t_w2c).T
        Xc_coco = Xc_coco.reshape(Nf, Nj, -1)
        Xc_coco[~vis_mask_2d_3d_i] = np.nan

        X3d_c_coco.append(Xc_coco)

    X3d_c_coco = np.array(X3d_c_coco)

    return X3d_c_coco


def save_joint(save_dir, kpt_op, kpt_score_op, aid, pid, gid, CAMERAS):

    assert kpt_op.ndim == 4
    Nc, Nf, Nj = kpt_score_op.shape
    print(f"save_joint:{save_dir}")
    for i, cid in enumerate(CAMERAS):
        kpt_op_i = kpt_op[i]
        kpt_score_op_i = kpt_score_op[i]
        with open(
            os.path.join(save_dir, f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"),
            "w",
        ) as fp:
            data = []
            for f, (p, s) in enumerate(zip(kpt_op_i, kpt_score_op_i)):
                # mask = ~np.isnan(p).all(axis=1)
                # s = np.ones(joint_j.shape[0])
                data.append(
                    {
                        "frame_index": int(f + 1),
                        "skeleton": [
                            {
                                "pose": p.flatten().astype(np.float64).tolist(),
                                "score": s.astype(np.float64).tolist(),
                            }
                        ],
                    }
                )
            json.dump({"data": data}, fp, indent=2, ensure_ascii=True)


def save_skelton_w_op(X3d_w_op, dst, gid):
    print(f"save_skelton_w_op:{dst}")
    Nf, Nj, _ = X3d_w_op.shape

    frames = np.arange(1, Nf + 1)

    with open(os.path.join(dst, f"skeleton_w_G{gid:03d}.json"), "w") as fp:
        out = {
            "skeleton": X3d_w_op.astype(np.float64).tolist(),
            "frame_indices": frames.tolist(),
        }
        json.dump(out, fp, indent=2, ensure_ascii=True)


def convert_coco_to_op(kp_coco):

    kp_op = np.full((kp_coco.shape[0], 25, kp_coco.shape[2]), np.nan, dtype=np.float32)
    for k_op, k_coco in op_to_coco.items():
        kp_op[:, OP_KEY[k_op], :] = kp_coco[:, COCO_KEY[k_coco], :]
    kp_op[:, OP_KEY["Neck"], :] = (
        kp_coco[:, COCO_KEY["L_Shoulder"], :] + kp_coco[:, COCO_KEY["R_Shoulder"], :]
    ) / 2
    kp_op[:, OP_KEY["MidHip"], :] = (
        kp_coco[:, COCO_KEY["R_Hip"], :] + kp_coco[:, COCO_KEY["L_Hip"], :]
    ) / 2
    return kp_op


def convert_coco_to_op_score(kp_coco):

    kp_op = np.full((kp_coco.shape[0], 25), np.nan, dtype=np.float32)
    for k_op, k_coco in op_to_coco.items():
        kp_op[:, OP_KEY[k_op]] = kp_coco[:, COCO_KEY[k_coco]]
    kp_op[:, OP_KEY["Neck"]] = (
        kp_coco[:, COCO_KEY["L_Shoulder"]] + kp_coco[:, COCO_KEY["R_Shoulder"]]
    ) / 2
    kp_op[:, OP_KEY["MidHip"]] = (
        kp_coco[:, COCO_KEY["R_Hip"]] + kp_coco[:, COCO_KEY["L_Hip"]]
    ) / 2
    return kp_op


def project_cv2(Rs, ts, Ks, X, width, height):

    assert Rs.ndim == 3
    assert Ks.ndim == 3
    assert ts.ndim == 3  # (C,3,1)

    Nc, _, _ = Rs.shape
    Nf, Nj, _ = X.shape
    X = X.reshape(Nf * Nj, 3)
    x_out = []
    for R, t, K in zip(Rs, ts, Ks):
        rvec = cv2.Rodrigues(R)[0]
        x, _ = cv2.projectPoints(X[None, :, :], rvec, t, K, np.zeros(0))
        x = x[:, -1, :]
        x[np.any(x < 0, axis=1), :] = np.nan
        x[x[:, 0] > width, :] = np.nan
        x[x[:, 1] > height, :] = np.nan
        x_out.append(x)
    x_out = np.array(x_out)
    x_out = x_out.reshape(Nc, Nf, Nj, 2)
    return np.array(x_out)


def get_poses_from_vp3d(third_party_dir, file_poses, file_scores, nframe):
    x2d_coco = []
    scores_coco_all = []

    for fpose, fscore in zip(file_poses, file_scores):
        # fid = fpose.split("/")[-1].split(".")[0]
        fid = fscore.split("/")[-1].split(".")[0]

        f_pose_npz = np.load(fpose, allow_pickle=True)

        x2d = f_pose_npz["positions_2d"].item()[f"{fid}.mp4"]["custom"][0]
        x2d_coco.append(x2d[:nframe, :, :])

        f_score_npz = np.load(fscore, allow_pickle=True)
        scores_coco = []
        for i in range(f_score_npz["keypoints"].shape[0]):
            scores_coco.append((f_score_npz["keypoints"][i][1][-1, :].T)[:, 3])
        scores_coco = np.array(scores_coco)
        scores_coco_all.append(scores_coco[:nframe, :])
    x2d_coco = np.array(x2d_coco)
    scores_coco_all = np.array(scores_coco_all)

    return x2d_coco, scores_coco_all


def convert_h36m_to_op(X_h36m_32):

    X_h36m_17 = X_h36m_32[:, list(H36M32_KEY.values()), :]
    X_op_25 = h36m_17_to_op(X_h36m_17)

    return X_op_25


def h36m_17_to_op(pose_h36m):
    J_open = 25
    T, _, N_COORD = pose_h36m.shape
    # check_a(pose_h36m, (T, J_h36m_17, None), None)
    pose_op = np.full((T, J_open, N_COORD), np.nan, dtype=pose_h36m.dtype)
    for k, op_idx in OP_KEY.items():
        if k in H36M17_KEY:
            pose_op[:, op_idx, :] = pose_h36m[:, H36M17_KEY[k], :]
        pose_op[:, OP_KEY["Neck"], :] = (
            pose_h36m[:, H36M17_KEY["RShoulder"], :]
            + pose_h36m[:, H36M17_KEY["LShoulder"], :]
        ) / 2.0
        pose_op[:, OP_KEY["MidHip"], :] = (
            pose_h36m[:, H36M17_KEY["RHip"], :] + pose_h36m[:, H36M17_KEY["LHip"], :]
        ) / 2.0

    return pose_op
