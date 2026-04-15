"""Per-frame joint visibility / orientation filtering used by the linear calibration."""
import numpy as np


def joints2orientations(p3d_CxNxJx3, mask_CxNxJ, bones_Jx2):
    C = p3d_CxNxJx3.shape[0]
    N = p3d_CxNxJx3.shape[1]
    J = p3d_CxNxJx3.shape[2]
    B = bones_Jx2.shape[0]
    assert p3d_CxNxJx3.shape[3] == 3
    assert mask_CxNxJ.shape == (C, N, J)
    assert bones_Jx2.shape[1] == 2
    assert mask_CxNxJ.dtype == bool

    p3d_CxNxJx3 = np.copy(p3d_CxNxJx3)

    # fill occluded joints by NaN
    p3d_CxNxJx3[~mask_CxNxJ] = np.nan

    # endpoints of each bone
    pairs = p3d_CxNxJx3[:, :, bones_Jx2, :]

    # e1 - e0
    dirs = pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]
    assert dirs.shape[-1] == 3

    # dirs.shape == Ndirs * 3
    dirs = dirs.reshape((C, N * B, 3))

    # delete dirs with NaN
    mask = np.min(~np.isnan(dirs), axis=(0, 2))
    dirs = dirs[:, mask, :]

    # normalize
    norm = np.linalg.norm(dirs, axis=2)
    assert np.all(norm > 0)
    dirs = dirs / norm[:, :, None]

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
    assert mask_CxNxJ.dtype == bool

    # fill occluded joints by NaN
    p2d_CxNxJx2[~mask_CxNxJ] = np.nan

    # select target joints
    p2d = p2d_CxNxJx2.reshape((C, -1, 2))
    idx = np.isnan(p2d).any(axis=(0, 2))
    p2d = p2d[:, ~idx, :]

    # verify
    assert p2d.shape[0] == C
    assert p2d.shape[2] == 2

    return p2d
