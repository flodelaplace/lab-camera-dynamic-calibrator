"""Pose JSON input/output: load and save 2D/3D keypoints + camera files."""
import functools
import json
import os

import numpy as np


def load_poses(filename):
    with open(filename, "r") as fp:
        P = json.load(fp)

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

    CAMID = np.array(cameras["CAMID"], dtype=int)
    K = np.array(cameras["K"], dtype=np.float64)
    R_w2c = np.array(cameras["R_w2c"], dtype=np.float64)
    t_w2c = np.array(cameras["t_w2c"], dtype=np.float64)
    # dist_coeffs is optional (backward compatible with old JSON files)
    if "dist_coeffs" in cameras:
        dist_coeffs = np.array(cameras["dist_coeffs"], dtype=np.float64)
    else:
        dist_coeffs = np.zeros((len(CAMID), 5), dtype=np.float64)
    assert len(CAMID) == len(K)
    assert len(CAMID) == len(R_w2c)
    assert len(CAMID) == len(t_w2c)

    return CAMID, K, R_w2c, t_w2c, dist_coeffs


def load_eldersim_skeleton_w(filename):
    with open(filename, "r") as fp:
        skeleton = json.load(fp)
        p3d_w = np.array(skeleton["skeleton"], dtype=np.float64)
        frames = np.array(skeleton["frame_indices"], dtype=int)

    assert len(p3d_w) == len(frames)

    return p3d_w, frames


def load_eldersim(dirname, gid, aid, pid, joint2d_dir="2d_joint"):
    CAMID, K, R_w2c, t_w2c, _dist = load_eldersim_camera(
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
        n_joints_2d = s2d.shape[1]  # detect from score array
        f2d_all.append(f2d)
        p2d_all.append(p2d.reshape((-1, n_joints_2d, 2)))
        s2d_all.append(s2d)

        f3d, p3d, s3d = load_poses(
            os.path.join(
                dirname, "3d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"
            )
        )
        n_joints_3d = s3d.shape[1]
        f3d_all.append(f3d)
        p3d_all.append(p3d.reshape((-1, n_joints_3d, 3)))
        s3d_all.append(s3d)

        # Use np.array_equal instead of np.alltrue to prevent deprecation issues
        assert np.array_equal(f2d, f3d)

    # check frames
    f2d_common = functools.reduce(np.intersect1d, [frames, *f2d_all])

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
