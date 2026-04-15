#%%
import os, sys
# Add repo root to import path so the core/ package and argument.py remain importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
from matplotlib.style import available
import numpy as np
import json
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from numba import jit
import time
from tqdm import tqdm
import core
from argument import parse_args
import matplotlib
matplotlib.use("Agg") # Mode sans interface graphique pour éviter les bugs sous WSL
import matplotlib.pyplot as plt
from pycalib.calib import *
import yaml
from core import project_cv2


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


# Note: a per-camera vs triangulated 3D consistency term (objfun_multiview3d)
# was previously prototyped here. It is intentionally NOT included: with MeTRAbs
# the per-camera 3D is too noisy frame-to-frame and conflicts with the 2D
# reprojection objective, which degrades the final MRE. Do not re-introduce.


def objfun(params, K, sp2d, ss2d, sp3d, ss3d, bone_idx, C, N, J, lambda1, lambda2, invalid_mask, conf_threshold=0.5):
    R_w2c, t_w2c, x = from_theta(params, C)
    E = []
    e = objfun_nll(
        K,
        R_w2c,
        t_w2c,
        x,
        sp2d,
        (ss2d > conf_threshold).reshape((C, N * J)),
        ss2d.reshape((C, N * J)),
    )
    E.append(e.flatten())

    e = objfun_var3d(R_w2c, sp3d, (ss3d > 0), bone_idx)
    E.append(e * lambda1)

    e = objfun_varbone(x.reshape(N, J, 3), bone_idx, invalid_mask)
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


def _save_cost_plot(cost_history, output_path, elapsed_secs=None, eval_count=0):
    """Save a snapshot of the cost convergence curve."""
    if len(cost_history) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cost_history, color='#2196F3', linewidth=1.5, marker='.', markersize=2)
    ax.set_yscale('log')
    ax.set_xlabel('Improvement steps')
    ax.set_ylabel('Cost (log scale)')
    title = f'BA Convergence — {eval_count} evals'
    if elapsed_secs:
        m, s = divmod(int(elapsed_secs), 60)
        title += f' — {m}m{s:02d}s'
    if len(cost_history) >= 2:
        reduction = cost_history[0] / cost_history[-1]
        title += f' — {reduction:.1f}x reduction'
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    fig.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def build_jac_sparsity(C, N, J, ss2d_work, ss3d, bone_idx, invalid_mask, conf_threshold):
    """Build Jacobian sparsity pattern for the BA problem.

    Tells scipy which parameters affect which residuals, avoiding full
    dense finite-difference Jacobian computation. Typically 50-200x speedup.
    """
    from scipy.sparse import coo_matrix

    n_cam = 6 * C
    n_pts = 3 * N * J
    n_params = n_cam + n_pts

    bone_idx_arr = np.array(bone_idx)
    B = len(bone_idx_arr)

    mask_nll = (ss2d_work > conf_threshold).reshape(C, N * J)

    # var3d residual count
    vc_mask = ss3d > 0  # C x N x J
    bv = vc_mask[:, :, bone_idx_arr[:, 0]] & vc_mask[:, :, bone_idx_arr[:, 1]]
    m_inv_v3d = ~bv.reshape(C, -1).all(axis=0)
    n_var3d = int((~m_inv_v3d).sum())

    n_varbone = B

    n_nll = 2 * int(mask_nll.sum())
    n_residuals = n_nll + n_var3d + n_varbone

    all_rows = []
    all_cols = []
    row = 0

    # --- NLL (vectorized per camera) ---
    for c in range(C):
        vis = np.where(mask_nll[c])[0]
        n_vis = len(vis)
        if n_vis == 0:
            continue
        cam_cols = np.concatenate([np.arange(3*c, 3*c+3),
                                   np.arange(3*C+3*c, 3*C+3*c+3)])  # (6,)
        cols_per_pt = np.concatenate([
            np.broadcast_to(cam_cols, (n_vis, 6)),
            n_cam + 3 * vis[:, None] + np.arange(3)
        ], axis=1)  # (n_vis, 9)

        even_rows = 2 * np.arange(n_vis) + row
        all_rows.append(np.repeat(even_rows, 9))
        all_rows.append(np.repeat(even_rows + 1, 9))
        all_cols.append(cols_per_pt.flatten())
        all_cols.append(cols_per_pt.flatten())
        row += 2 * n_vis

    # --- var3d (depends on all camera rvecs) ---
    if n_var3d > 0:
        rvec_cols = np.arange(3 * C)
        all_rows.append(np.repeat(np.arange(n_var3d) + row, 3 * C))
        all_cols.append(np.tile(rvec_cols, n_var3d))
        row += n_var3d

    # --- varbone (depends on bone endpoint 3D coords across all frames) ---
    for b in range(B):
        j1, j2 = bone_idx_arr[b]
        fi = np.arange(N)
        pt_cols = np.concatenate([
            (n_cam + 3 * (fi * J + j1)[:, None] + np.arange(3)).flatten(),
            (n_cam + 3 * (fi * J + j2)[:, None] + np.arange(3)).flatten()
        ])
        all_rows.append(np.full(len(pt_cols), row))
        all_cols.append(pt_cols)
        row += 1

    assert row == n_residuals, f"Sparsity row mismatch: {row} vs {n_residuals}"

    r = np.concatenate(all_rows) if all_rows else np.array([], dtype=int)
    c = np.concatenate(all_cols) if all_cols else np.array([], dtype=int)
    S = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)),
                   shape=(n_residuals, n_params))
    density = S.nnz / max(n_residuals * n_params, 1)
    print(f"  Jacobian sparsity: {n_residuals} residuals x {n_params} params, "
          f"density={density:.6f}, nnz={S.nnz}")
    return S.tocsc()


def _run_ba(K, R_w2c, t_w2c, x_all, sp2d_flat, ss2d, sp3d, ss3d, bone_idx,
            C, N, J, lambda1, lambda2, invalid_mask, conf_threshold, cost_history,
            plot_path=None, jac_sparsity=None):
    """Single pass of bundle adjustment optimization."""
    best_cost = cost_history[-1] if cost_history else float('inf')
    pbar = tqdm(desc="  BA optimizing", unit="eval", dynamic_ncols=True)
    eval_count = [0]
    start_time = time.time()
    last_plot_time = [start_time]

    def objfun_wrapped(params, *args):
        nonlocal best_cost
        E = objfun(params, *args)
        cost = 0.5 * np.sum(E**2)
        eval_count[0] += 1
        if cost < best_cost:
            best_cost = cost
            cost_history.append(cost)
        pbar.update(1)
        elapsed = time.time() - start_time
        m, s = divmod(int(elapsed), 60)
        pbar.set_postfix(cost=f"{best_cost:.2f}", time=f"{m}m{s:02d}s", refresh=False)
        # Save live cost plot every 30s
        if plot_path and (time.time() - last_plot_time[0]) > 10:
            _save_cost_plot(cost_history, plot_path, elapsed, eval_count[0])
            last_plot_time[0] = time.time()
        return E

    theta0 = to_theta(R_w2c, t_w2c, x_all)

    # Scale max_nfev by problem size (more params = more evals needed for sparsity)
    n_params = len(theta0)
    max_evals = min(max(60000, 4 * n_params), 80000)
    kwargs = dict(
        verbose=0,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        max_nfev=max_evals,
        method="trf",
        args=(K, sp2d_flat, ss2d, sp3d, ss3d, bone_idx,
              C, N, J, lambda1, lambda2, invalid_mask, conf_threshold),
    )
    if jac_sparsity is not None:
        kwargs['jac_sparsity'] = jac_sparsity

    res = least_squares(objfun_wrapped, theta0, **kwargs)

    pbar.close()
    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"  BA converged in {eval_count[0]} evaluations, {m}m{s:02d}s "
          f"(final cost: {best_cost:.2f})")
    if plot_path:
        _save_cost_plot(cost_history, plot_path, elapsed, eval_count[0])
        print(f"  Cost curve: {plot_path}")

    return from_theta(res["x"], C)


def ba_main(camid, K, R_w2c, t_w2c, sp2d, ss2d, sp3d, ss3d, lambda1, lambda2,
            conf_threshold=0.5, bone_idx=None, n_iterations=2, outlier_threshold=2.0,
            plot_dir=None):

    C = len(camid)
    N = sp2d.shape[1]
    J = sp2d.shape[2]

    if bone_idx is None:
        bone_idx = core.OP_BONE

    cost_history = []
    ss2d_work = ss2d.copy()

    for iteration in range(n_iterations):
        print(f"\n{'='*50}")
        print(f"  BA Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*50}")

        # Triangulate 3D points
        x_all = core.triangulate_with_conf(sp2d, ss2d_work, K, R_w2c, t_w2c, (ss2d_work > conf_threshold))
        x_all = x_all.reshape(N * J, 3)
        assert x_all.shape == (N * J, 3)

        # Replace NaNs for optimizer
        invalid_mask = np.isnan(x_all).any(axis=1)
        x_all[invalid_mask] = 0.0

        # Zero out confidence for invalid 3D points
        invalid_mask_3d = invalid_mask.reshape(N, J)
        for c in range(C):
            ss2d_work[c][invalid_mask_3d] = 0.0

        sp2d_flat = sp2d.reshape((C, N * J, 2))

        e_nll = objfun_nll(K, R_w2c, t_w2c, x_all, sp2d_flat,
                       (ss2d_work > conf_threshold).reshape((C, N * J)),
                       ss2d_work.reshape((C, N * J)))
        nll_energy = np.sum(e_nll**2)
        print(f"  Mean NLL: {np.mean(e_nll**2):.4f}")

        e_v3d = objfun_var3d(R_w2c, sp3d, (ss3d > 0), bone_idx)
        print(f"  Mean 3D variance: {np.mean(e_v3d**2):.6f}")

        e_bone = objfun_varbone(x_all.reshape(N, J, 3), bone_idx, invalid_mask)
        bone_energy = np.sum(e_bone**2)
        print(f"  Mean bone-length variance: {np.mean(e_bone**2):.6f}")

        # Auto-balance lambda2: bone term should be ~10% of NLL
        # If bone variance is negligible (e.g. with MeTRAbs metric 3D), disable
        # the bone term to avoid runaway lambda2 and wasted optimization time.
        if bone_energy > 1e-3:
            target_ratio = 0.1
            lambda2 = np.sqrt(target_ratio * nll_energy / bone_energy)
            # Cap to avoid extreme values when bone_energy is tiny
            lambda2 = min(lambda2, 1000.0)
            print(f"  Auto-balanced lambda2: {lambda2:.4f} "
                  f"(NLL={nll_energy:.0f}, bone={bone_energy:.0f})")
        else:
            lambda2 = 0.0
            print(f"  Bone variance negligible ({bone_energy:.6f}), "
                  f"disabling bone regularization (lambda2=0)")

        # Build Jacobian sparsity pattern (huge speedup for BA)
        print("  Building Jacobian sparsity pattern...")
        t0 = time.time()
        jac_sp = build_jac_sparsity(C, N, J, ss2d_work, ss3d, bone_idx,
                                    invalid_mask, conf_threshold)
        print(f"  Sparsity built in {time.time()-t0:.1f}s")

        # Verify residual count matches
        theta_test = to_theta(R_w2c, t_w2c, x_all)
        r_test = objfun(theta_test, K, sp2d_flat, ss2d_work, sp3d, ss3d,
                        bone_idx, C, N, J, lambda1, lambda2, invalid_mask,
                        conf_threshold)
        if jac_sp.shape[0] != len(r_test):
            print(f"  WARNING: Sparsity rows ({jac_sp.shape[0]}) != residuals "
                  f"({len(r_test)}), falling back to dense Jacobian")
            jac_sp = None

        plot_path = (os.path.join(plot_dir, f"ba_cost_live_iter{iteration+1}.png")
                     if plot_dir else None)

        # Run optimization
        R_w2c, t_w2c, x_opt = _run_ba(
            K, R_w2c, t_w2c, x_all, sp2d_flat, ss2d_work, sp3d, ss3d,
            bone_idx, C, N, J, lambda1, lambda2, invalid_mask, conf_threshold,
            cost_history, plot_path=plot_path, jac_sparsity=jac_sp
        )

        # Outlier rejection after all but the last iteration
        if iteration < n_iterations - 1:
            # Compute per-frame reprojection error
            x_tri = x_opt.reshape(N, J, 3)
            frame_errors = np.zeros(N)
            for f in range(N):
                errs = []
                for c in range(C):
                    mask_f = ss2d_work[c, f, :] > conf_threshold
                    if mask_f.sum() == 0:
                        continue
                    pts3d = x_tri[f, mask_f, :]
                    pts2d_obs = sp2d[c, f, mask_f, :]
                    proj = K[c] @ (R_w2c[c] @ pts3d.T + t_w2c[c].reshape(3, 1))
                    proj = (proj[:2] / proj[2]).T
                    errs.append(np.mean(np.linalg.norm(pts2d_obs - proj, axis=1)))
                frame_errors[f] = np.mean(errs) if errs else 0

            median_err = np.median(frame_errors[frame_errors > 0])
            outlier_mask = frame_errors > outlier_threshold * median_err
            n_outliers = np.sum(outlier_mask)

            if n_outliers > 0:
                print(f"\n  Outlier rejection: {n_outliers}/{N} frames removed "
                      f"(threshold: {outlier_threshold:.1f}x median={median_err:.2f}px)")
                # Zero out confidence for outlier frames
                ss2d_work = ss2d.copy()
                for c in range(C):
                    ss2d_work[c][invalid_mask_3d] = 0.0
                    ss2d_work[c, outlier_mask, :] = 0.0
            else:
                print(f"\n  No outliers found (median error: {median_err:.2f}px)")

    return R_w2c, t_w2c, x_opt, cost_history


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


def save_mask(intrinsic, R, t, obs_mask, width, height, conf_threshold=0.5):

    # if bObsMask:
    print("save obs. mask")
    sCAMID_all, _, _, _, _, _, _, sp2d_all, ss2d_all, sframes_all = core.load_eldersim(
        PREFIX, GID, AID, PID
    )
    x_all = core.triangulate_with_conf(
        sp2d_all, ss2d_all, intrinsic, R, t, (ss2d_all > conf_threshold)
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

    with open(os.path.join(_REPO_ROOT, "config", "config.yaml")) as file:
        config = yaml.safe_load(file.read())

    width = config[DATASET]["width"]
    height = config[DATASET]["height"]
    available_joints = config[DATASET]["available_joints"]
    FRAME_SKIP = args.frame_skip

    # if OBS_MASK:
    # sCAMID, _ , _, _, sp3d_w, sp3d, ss3d, sp2d, ss2d, sframes = core.load_eldersim(PREFIX, GID, AID, PID,joint2d_dir='2d_joint_mask')
    #     print("12323132")
    # else :
    sCAMID, _, _, _, sp3d_w, sp3d, ss3d, sp2d, ss2d, sframes = core.load_eldersim(
        PREFIX, GID, AID, PID
    )

    CAMID, intrinsic, R_w2c, t_w2c, _dist_calib = core.load_eldersim_camera(JSON_IN)
    # Load dist_coeffs from the original cameras file (not the calib result)
    _cam_file = os.path.join(PREFIX, f"cameras_G{GID:03d}.json")
    if os.path.exists(_cam_file):
        _, _, _, _, dist_coeffs = core.load_eldersim_camera(_cam_file)
    else:
        dist_coeffs = np.zeros((len(CAMID), 5), dtype=np.float64)
    LAMBDA1 = args.ba_lambda1
    LAMBDA2 = args.ba_lambda2
    CONF_THRESHOLD = args.conf_threshold
    assert np.alltrue(CAMID == sCAMID)

    sp3d_w = sp3d_w[::FRAME_SKIP, :, :]
    sp3d = sp3d[:, ::FRAME_SKIP, :, :]
    ss3d = ss3d[:, ::FRAME_SKIP, :]
    sp2d = sp2d[:, ::FRAME_SKIP, available_joints, :]
    ss2d = ss2d[:, ::FRAME_SKIP, available_joints]
    sframes = sframes[::FRAME_SKIP]

    # Auto-detect skeleton based on joint count
    N_JOINTS = sp2d.shape[2]
    BONE_IDX, _ = core.get_bone_config(N_JOINTS)
    print(f"dataset={DATASET}")
    print(f"target BA={JSON_IN}")
    print(f"Detected {N_JOINTS} joints -> using {len(BONE_IDX)} bones")

    plot_dir = os.path.dirname(JSON_OUT)
    os.makedirs(plot_dir, exist_ok=True)

    R_w2c_opt, t_w2c_opt, x_opt, cost_history = ba_main(
        CAMID, intrinsic, R_w2c, t_w2c, sp2d, ss2d, sp3d, ss3d, LAMBDA1, LAMBDA2, CONF_THRESHOLD, BONE_IDX,
        plot_dir=plot_dir
    )

    # Génération et sauvegarde de la courbe d'optimisation
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='#2196F3', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Améliorations (Évaluations)')
    plt.ylabel('Erreur globale (Échelle Log)')
    plt.title(f'Courbe de convergence du Bundle Adjustment (Skip: {FRAME_SKIP})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    curve_path = JSON_OUT.replace('.json', '_curve.png')
    plt.savefig(curve_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"📈 Courbe d'optimisation sauvegardée : {curve_path}")

    if SAVE_OBS_MASK:
        save_mask(intrinsic, R_w2c_opt, t_w2c_opt, OBS_MASK, width, height, CONF_THRESHOLD)

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
