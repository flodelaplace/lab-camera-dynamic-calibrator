"""Multi-view geometry: triangulation, projection, R/T transformations.

Functions:
- triangulate / triangulate_point / triangulate_with_conf — DLT triangulation
- constraint_mat / constraint_mat_from_single_view — building blocks for triangulation
- project / project_cv2 — 3D → 2D projection
- invRT / invRT_batch — invert a w2c transformation to c2w
- z_test_w2c — cheirality (sign) test for two-view triangulation
"""
import cv2
import numpy as np


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


def triangulate(pt2d, P):
    """Triangulate a 3D point from two or more views by DLT."""
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
            X.append(x3d[:3])
    X = np.array(X)
    X = X.reshape(Nf, Nj, 3)
    return X


def project(K, R_w2c, t_w2c, pts3d_w):
    p = K @ (R_w2c @ pts3d_w.T + t_w2c[:, None])
    p = p / p[2, :]
    return p.T


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
