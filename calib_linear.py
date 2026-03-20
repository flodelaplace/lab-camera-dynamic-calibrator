#%%
import cv2
import numpy as np
import sys
import json
import itertools
import os
import scipy as sp

# from numba import jit
from argument import parse_args
from util import *
import pycalib

# module_path = os.path.abspath(os.path.join('./pycalib/'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# import pycalib


def collinearity_w2c(R_w2c, n, idx_v, idx_t, num_v, num_t):
    nmat = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

    t0 = num_v * 3
    # A = np.zeros((3, (num_v + num_t)*3))
    A = sp.sparse.lil_matrix((3, (num_v + num_t) * 3), dtype=np.float64)
    A[:, idx_v * 3 : idx_v * 3 + 3] = nmat @ R_w2c
    A[:, t0 + idx_t * 3 : t0 + idx_t * 3 + 3] = nmat

    return A


def coplanarity_w2c(Ra, Rb, na, nb, idx_t1, idx_t2, num_t):
    rows = na.shape[0]
    assert na.shape[1] == 3
    assert nb.shape[0] == rows
    assert nb.shape[1] == 3

    m = np.cross(na @ Ra, nb @ Rb)
    # A = np.zeros((rows, num_t * 3))
    A = sp.sparse.lil_matrix((rows, num_t * 3), dtype=np.float64)
    A[:, idx_t1 * 3 : idx_t1 * 3 + 3] = m @ Ra.T
    A[:, idx_t2 * 3 : idx_t2 * 3 + 3] = -m @ Rb.T
    return A


def calib_linear(v_CxNx3, n_CxMx3):
    C = v_CxNx3.shape[0]
    N = v_CxNx3.shape[1]
    M = n_CxMx3.shape[1]
    assert v_CxNx3.shape[2] == 3
    assert n_CxMx3.shape[0] == C
    assert n_CxMx3.shape[2] == 3

    # Rotation
    v_Nx3C = np.hstack(v_CxNx3)
    Y, D, Zt = np.linalg.svd(v_Nx3C)
    # print('ratio =', np.sum(D[:3])/np.sum(D))
    V = Y[:, :3] @ np.diag(D[:3]) / np.sqrt(C)
    R_all = np.sqrt(C) * Zt[:3, :]
    # make R0 be I (also correct handedness)
    Rx = np.linalg.inv(R_all[:3, :3])
    R_all = Rx @ R_all
    assert np.linalg.det(R_all[:3, :3]) > 0

    R_w2c_list = R_all.T.reshape((-1, 3, 3))

    # Translation
    A = []
    for idx_t, (R, n) in enumerate(zip(R_w2c_list, n_CxMx3)):
        for idx_v in range(n.shape[0]):
            A.append(collinearity_w2c(R, n[idx_v, :], idx_v, idx_t, M, C))
    # A = np.vstack(A)
    A = sp.sparse.vstack(A)

    B = []
    for ((a, Ra, na), (b, Rb, nb)) in itertools.combinations(
        zip(range(C), R_w2c_list, n_CxMx3), 2
    ):
        B.append(coplanarity_w2c(Ra, Rb, na, nb, a, b, C))
    # B = np.vstack(B)
    B = sp.sparse.vstack(B)

    C_mat = sp.sparse.lil_matrix((A.shape[0] + B.shape[0], A.shape[1]), dtype=np.float64)
    C_mat[: A.shape[0]] = A
    C_mat[A.shape[0] :, -B.shape[1] :] = B

    try:
        w, v = sp.linalg.eigh(
            (C_mat.T @ C_mat).toarray(), subset_by_index=(0, 5), overwrite_a=True, overwrite_b=True
        )
    except np.linalg.LinAlgError:
        print("WARN: SVD for calibration did not converge. Returning empty results.")
        return None, None, None

    if w[3] / w[4] > 1e-4:
        print(f"WARN: degenerate case (only 4 eigenvalues should be zero): lambda={w}")

    k = v[:, :4]

    _, s, vt = np.linalg.svd(k[-B.shape[1] : -B.shape[1] + 3, :])
    t = k @ vt[3, :].T
    X = t[: -B.shape[1]].reshape((-1, 3))
    t = t[-B.shape[1] :]
    s = np.linalg.norm(t[3:6])
    if s < 1e-9:
        print("WARN: Scale is close to zero. Calibration may be unstable.")
        return None, None, None
        
    t = t / s
    X = X / s
    t_w2c_list = t.reshape((-1, 3))

    R1, R2 = R_w2c_list[0], R_w2c_list[1]
    t1, t2 = t_w2c_list[0], t_w2c_list[1]
    n1, n2 = n_CxMx3[0], n_CxMx3[1]
    sign, Np, Nn = z_test_w2c(R1, t1, R2, t2, n1, n2)

    t_w2c_list = sign * t_w2c_list
    X = sign * X

    P = np.concatenate((R_w2c_list, t_w2c_list[:, :, None]), axis=2)
    X2 = np.array([pycalib.calib.triangulate(n_CxMx3[:, i, :], P)[:3] for i in range(M)])

    return R_w2c_list, t_w2c_list.reshape((-1, 3, 1)), X2


def main_linear(
    dirname, gid, aid, pid, bone_idx, joint_idx, bObs_mask, *, 
    frame_start=None, frame_end=None, frame_skip=1
):

    if bObs_mask:
        CAMID, K, _, _, _, p3d, s3d, p2d, s2d, frames = load_eldersim(
            dirname, gid, aid, pid, joint2d_dir="2d_joint_mask"
        )
    else:
        CAMID, K, _, _, _, p3d, s3d, p2d, s2d, frames = load_eldersim(
            dirname, gid, aid, pid
        )

    # --- Frame selection (chunking and skipping) ---
    if frame_start is not None and frame_end is not None:
        # Treat frame_end as inclusive for consistency with the pipeline CLI
        end_idx = frame_end + 1
        p3d = p3d[:, frame_start:end_idx, :, :]
        p2d = p2d[:, frame_start:end_idx, :, :]
        s3d = s3d[:, frame_start:end_idx, :]
        s2d = s2d[:, frame_start:end_idx, :]
    
    p3d = p3d[:, ::frame_skip, :, :]
    p2d = p2d[:, ::frame_skip, :, :]
    s3d = s3d[:, ::frame_skip, :]
    s2d = s2d[:, ::frame_skip, :]

    if p3d.shape[1] == 0:
        print("WARN: No frames left after chunking/skipping. Skipping this chunk.")
        return None, None, None, None, None, None

    mask_CxNxJ = (s2d > 0.5) * (s3d == 1) # Ignore les prédictions peu fiables (frames noires/floues)

    vc = joints2orientations(p3d, mask_CxNxJ, bone_idx)
    if vc.shape[1] == 0:
        print("WARN: No valid orientations found in this chunk. Skipping.")
        return None, None, None, None, None, None

    y = joints2projections(p2d, mask_CxNxJ, joint_idx)
    if y.shape[1] == 0:
        print("WARN: No valid 2D projections found in this chunk. Skipping.")
        return None, None, None, None, None, None

    n = np.ones((y.shape[0], y.shape[1], 3), dtype=np.float64)
    n[:, :, :2] = y
    
    ni = [n[i] @ np.linalg.inv(K[i]).T for i in range(len(K))]
    n = np.array(ni)

    print(f"Processing chunk: vc={vc.shape}, n={n.shape}")

    R_w2c_est, t_w2c_est, p3d_w_est = calib_linear(vc, n)
    
    if R_w2c_est is None:
        return None, None, None, None, None, None

    e = 0
    if p3d_w_est is not None and y.shape[1] > 0:
        E = []
        for k, R, t, pts2d in zip(K, R_w2c_est, t_w2c_est, y):
            p = k @ (R @ p3d_w_est.T + t)
            p = p / p[2, :]
            e_mat = pts2d - p[:2, :].T
            E.append(e_mat)
        e = np.mean(np.linalg.norm(np.vstack(E), axis=1))

    return R_w2c_est, t_w2c_est, p3d_w_est, e, CAMID, K


if __name__ == "__main__":
    args = parse_args()
    PREFIX = args.prefix + "/" + args.target
    AID = args.aid
    PID = args.pid
    GID = args.gid
    DATASET = args.dataset
    OBS_MASK = args.obs_mask
    
    # Use a specific output file if chunk_id is provided
    if args.chunk_id is not None:
        chunk_dir = os.path.join(args.prefix, "results", "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        JSON_OUT = os.path.join(chunk_dir, f"linear_chunk_{args.chunk_id}.json")
    else:
        os.makedirs(args.prefix + "/results/", exist_ok=True)
        JSON_OUT = os.path.join(args.prefix, "results", f"linear_{args.target.split('_')[1]}_{args.target.split('_')[2]}.json")

    print(f"dataset={DATASET}")
    
    R, t, X, e, CAMID, K = main_linear(
        PREFIX, GID, AID, PID, OP_BONE, OP_KEY_SUB, OBS_MASK, 
        frame_start=args.frame_start, 
        frame_end=args.frame_end,
        frame_skip=args.frame_skip
    )

    if R is not None:
        with open(JSON_OUT, "w") as fp:
            out = {
                "CAMID": CAMID.tolist(),
                "K": K.tolist(),
                "R_w2c": R.tolist(),
                "t_w2c": t.tolist(),
            }
            json.dump(out, fp, indent=2, ensure_ascii=True)
        print(f"Saved calibration to {JSON_OUT}")
        print(f"Reprojection error: {e if e is not None else 'N/A'}")
    else:
        print("Calibration failed for this chunk, no file saved.")

    print(" ")
