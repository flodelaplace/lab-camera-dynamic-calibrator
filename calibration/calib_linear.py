#%%
import os, sys
# Add repo root to import path so the core/ package and argument.py remain importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import json
import itertools
import scipy as sp

# from numba import jit
from argument import parse_args
from core import *
from core import get_bone_config
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


def procrustes_align(X_src, X_tgt):
    """Find R, t, s such that X_tgt ≈ s * R @ X_src + t (Umeyama method).

    Args:
        X_src: (N, 3) source points
        X_tgt: (N, 3) target points
    Returns:
        R (3,3), t (3,), s (float)
    """
    n = X_src.shape[0]
    mu_src = X_src.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)
    X_src_c = X_src - mu_src
    X_tgt_c = X_tgt - mu_tgt

    var_src = np.sum(X_src_c ** 2) / n

    H = X_src_c.T @ X_tgt_c / n
    U, D, Vt = np.linalg.svd(H)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = Vt.T @ S @ U.T
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_tgt - s * R @ mu_src

    return R, t, s


def calib_procrustes(p3d_CxNxJx3, s3d_CxNxJ, K_all, p2d_CxNxJx2, s2d_CxNxJ, conf_threshold=0.5):
    """Estimate extrinsic calibration using Procrustes alignment of per-camera 3D poses.

    Uses camera 0 as reference frame. For each other camera, aligns its 3D skeleton
    to camera 0's skeleton to get relative R, t.

    Returns:
        R_w2c (C, 3, 3), t_w2c (C, 3, 1), X_world (M, 3)
    """
    C, N, J, _ = p3d_CxNxJx3.shape
    mask = s3d_CxNxJ > 0  # (C, N, J)

    # Reference camera is camera 0 — its 3D defines the world frame
    R_w2c_list = [np.eye(3)]
    t_w2c_list = [np.zeros(3)]

    ref_3d = p3d_CxNxJx3[0]  # (N, J, 3) — reference camera's 3D

    for c in range(1, C):
        cam_3d = p3d_CxNxJx3[c]  # (N, J, 3)

        # Find frames+joints visible in both cameras
        joint_mask = mask[0] & mask[c]  # (N, J)
        valid_idx = np.where(joint_mask.flatten())[0]

        if len(valid_idx) < 10:
            print(f"  WARN: Camera {c} — only {len(valid_idx)} shared points, using identity")
            R_w2c_list.append(np.eye(3))
            t_w2c_list.append(np.zeros(3))
            continue

        pts_ref = ref_3d.reshape(-1, 3)[valid_idx]
        pts_cam = cam_3d.reshape(-1, 3)[valid_idx]

        # Procrustes: find R, t, s such that pts_cam ≈ s * R @ pts_ref + t
        # This gives us R_cam_from_ref and t_cam_from_ref
        R, t, s = procrustes_align(pts_ref, pts_cam)

        # R_w2c[c] = R (world=ref frame -> camera c frame)
        R_w2c_list.append(R)
        t_w2c_list.append(t)

        residual = np.mean(np.linalg.norm(pts_cam - (s * (R @ pts_ref.T).T + t), axis=1))
        print(f"  Camera {c}: Procrustes residual = {residual:.2f}mm, scale = {s:.4f}")

    R_w2c = np.array(R_w2c_list)
    t_w2c = np.array(t_w2c_list).reshape(-1, 3, 1)

    # Triangulate world points using the estimated extrinsics
    joint_mask_2d = s2d_CxNxJ > conf_threshold
    P_list = []
    for c in range(C):
        P_list.append(K_all[c] @ np.hstack([R_w2c[c], t_w2c[c]]))

    X_world = []
    for f in range(N):
        for j in range(J):
            pts2d = p2d_CxNxJx2[:, f, j, :]
            vis = joint_mask_2d[:, f, j]
            if vis.sum() >= 2:
                x3d = pycalib.calib.triangulate(pts2d[vis], np.array(P_list)[vis])[:3]
            else:
                x3d = np.full(3, np.nan)
            X_world.append(x3d)
    X_world = np.array(X_world).reshape(N * J, 3)

    return R_w2c, t_w2c, X_world


def main_linear(
    dirname, gid, aid, pid, bone_idx, joint_idx, bObs_mask, *, 
    frame_start=None, frame_end=None, frame_skip=1, conf_threshold=0.5
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

    N_after = p3d.shape[1]
    n_joints = p3d.shape[2]

    if N_after == 0:
        print("WARN: No frames left after chunking/skipping. Skipping this chunk.")
        return None, None, None, None, None, None

    mask_CxNxJ = (s2d > conf_threshold) * (s3d > conf_threshold) # Ignore les prédictions peu fiables (frames noires/floues)

    # --- Visibility-based frame selection ---
    # Only keep frames where the person is visible from enough cameras
    C = p3d.shape[0]
    frame_vis = mask_CxNxJ.any(axis=2).sum(axis=0)  # (N,) cameras seeing person per frame
    min_cams = max(2, (C * 2 + 2) // 3)  # >= 2/3 of cameras (rounded up)

    good = frame_vis >= min_cams
    if good.sum() < 20:
        # Fallback: lower threshold if too few frames
        min_cams = 2
        good = frame_vis >= min_cams

    n_dropped = N_after - int(good.sum())
    if n_dropped > 0:
        print(f"  Visibility filter: {int(good.sum())}/{N_after} frames "
              f"(>= {min_cams}/{C} cameras)")
        p3d = p3d[:, good, :, :]
        p2d = p2d[:, good, :, :]
        s3d = s3d[:, good, :]
        s2d = s2d[:, good, :]
        mask_CxNxJ = mask_CxNxJ[:, good, :]
        N_after = int(good.sum())

    if N_after == 0:
        print("WARN: No frames left after visibility filter. Skipping.")
        return None, None, None, None, None, None

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

    # Try Procrustes first if we have good 3D data (MeTRAbs, 26 joints)
    n_joints = p3d.shape[2]
    use_procrustes = (n_joints == 26 and np.any(s3d > 0))

    if use_procrustes:
        print("  Using Procrustes initialization (MeTRAbs 3D available)")
        R_w2c_est, t_w2c_est, p3d_w_est = calib_procrustes(
            p3d, s3d, K, p2d, s2d, conf_threshold
        )
        p3d_w_est_flat = p3d_w_est.reshape(-1, 3)
        # Remove NaN rows for reprojection error calculation
        valid_3d = ~np.isnan(p3d_w_est_flat).any(axis=1)
    else:
        R_w2c_est, t_w2c_est, p3d_w_est = calib_linear(vc, n)

    if R_w2c_est is None:
        return None, None, None, None, None, None

    e = 0
    if use_procrustes and p3d_w_est is not None:
        # Procrustes path: X_world is (N*J, 3), use full p2d/s2d for MRE
        X_w = p3d_w_est.reshape(N_after, n_joints, 3)
        E = []
        for c in range(len(K)):
            for f in range(N_after):
                for j in range(n_joints):
                    if s2d[c, f, j] > conf_threshold and not np.isnan(X_w[f, j]).any():
                        pt3d = X_w[f, j]
                        proj = K[c] @ (R_w2c_est[c] @ pt3d + t_w2c_est[c].flatten())
                        proj = proj[:2] / proj[2]
                        obs = p2d[c, f, j, :]
                        E.append(np.linalg.norm(obs - proj))
        e = np.mean(E) if E else 999.0
    elif p3d_w_est is not None and y.shape[1] > 0:
        # Linear path: p3d_w_est matches y (filtered visible points)
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
    
    # Auto-detect skeleton: peek at a 3D joint file to get the number of joints
    import glob as _glob
    _j3d_files = sorted(_glob.glob(os.path.join(PREFIX, "3d_joint", f"A{AID:03d}_P{PID:03d}_G{GID:03d}_C*.json")))
    if _j3d_files:
        with open(_j3d_files[0]) as _f:
            _n_joints = len(json.load(_f)["data"][0]["skeleton"][0]["score"])
    else:
        _n_joints = 25
    _bone_idx, _key_sub = get_bone_config(_n_joints)
    print(f"Detected {_n_joints} joints -> using {len(_bone_idx)} bones")

    R, t, X, e, CAMID, K = main_linear(
        PREFIX, GID, AID, PID, _bone_idx, _key_sub, OBS_MASK,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_skip=args.frame_skip,
        conf_threshold=args.conf_threshold
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
