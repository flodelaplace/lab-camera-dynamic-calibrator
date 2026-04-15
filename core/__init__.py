"""Shared library for the calibration pipeline.

Submodules:
- skeletons : joint indices, bone connectivity (OP, MeTRAbs-26, bml_movi_87)
- geometry  : triangulation, projection, R/T inversion
- poses_io  : JSON load/save for poses, cameras, skeletons
- filtering : per-frame visibility / orientation helpers used by linear calibration
- gpu       : GPU selection helper

Everything is re-exported here so consumers can do either:
    from core import OP_BONE, load_poses, triangulate_with_conf
or:
    from core.skeletons import OP_BONE
    from core.poses_io import load_poses
"""
from .skeletons import (
    OP_KEY, COCO_KEY, H36M32_KEY, H36M17_KEY, op_to_coco,
    OP_BONE, OP_KEY_SUB, OP_BONE_SUB, mk_bone_sub,
    METRABS_KEY, METRABS_BML87_INDICES, MK,
    METRABS_BONE, METRABS_KEY_SUB, METRABS_BONE_SUB,
    BML87_KEY, B, BML87_BONE, BML87_KEY_SUB, BML87_BONE_SUB,
    get_bone_config,
)
from .geometry import (
    z_test_w2c, triangulate, constraint_mat_from_single_view, constraint_mat,
    triangulate_point, triangulate_with_conf,
    project, project_cv2, invRT, invRT_batch,
)
from .poses_io import (
    load_poses, load_eldersim_camera, load_eldersim_skeleton_w, load_eldersim,
    save_cam, save_joint,
)
from .filtering import joints2orientations, joints2projections
from .gpu import select_gpu
