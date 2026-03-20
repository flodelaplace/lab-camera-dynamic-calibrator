import argument
import os, sys
import numpy as np
from argument import parse_args
from pycalib.calib import rebase_all
from util import (
    COCO_KEY,
    H36M32_KEY,
    invRT_batch,
    load_eldersim_camera,
    load_poses,
    triangulate_with_conf,
)
from scipy.spatial.transform import Rotation
import yaml

vp3d_path = os.path.abspath(os.path.join("./third_party/VideoPose3D"))
if vp3d_path not in sys.path:
    sys.path.append(vp3d_path)
from data.data_utils import suggest_metadata

SCALE = 2.3


def savedata(
    canonical_name, R_w2c, t_w2c, x2d_gafa_coco, x3d_gafa_32, camids, width, height
):

    metadata = suggest_metadata("coco")
    metadata["video_metadata"] = {}
    video_metadata = {}
    video_metadata["w"] = width
    video_metadata["h"] = height
    output_2d = {}
    output_3d = {}
    output_2d[canonical_name] = {}
    output_3d[canonical_name] = {}
    keypoints = {}
    output_cameras = {}
    for camid, gp2d_i in zip(camids, x2d_gafa_coco):
        keypoints[camid] = gp2d_i
    R_c2w, t_c2w = invRT_batch(R_w2c, t_w2c)
    cams_gafa = create_cameras(R_c2w, t_c2w, width, height, camids)
    output_cameras[canonical_name] = cams_gafa
    output_2d[canonical_name]["custom"] = keypoints
    output_3d[canonical_name]["custom"] = x3d_gafa_32
    metadata["video_metadata"][canonical_name] = video_metadata
    return output_2d, output_3d, metadata, output_cameras


def create_cameras(R_c2w_gt, t_c2w_gt, res_w, res_h, gCAMID):
    cameras = {}
    N = R_c2w_gt.shape[0]
    for n, camid in enumerate(gCAMID):
        quat = Rotation.from_matrix(R_c2w_gt[n, :, :]).as_quat()
        orientation = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        translation = t_c2w_gt[n, :].astype(np.float32)

        cameras[camid] = {
            "orientation": orientation,
            "translation": translation,
            "res_w": res_w,
            "res_h": res_h,
        }
    return cameras


def load_cooco_poses_all(dirname, cameras, aid, pid, gid):
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
        p_all.append(p.reshape((N, 17, -1)))  # to work both for 2D and 3D
        s_all.append(s)

    return np.array(f_all), np.array(p_all), np.array(s_all)


def convert_coco_to_h36m32(kp_coco):
    kp_h36m_32 = np.full(
        (kp_coco.shape[0], 32, kp_coco.shape[2]), np.nan, dtype=np.float32
    )
    for coco_key, coco_value in COCO_KEY.items():
        if not (
            coco_key == "Nose"
            or coco_key == "L_Eye"
            or coco_key == "R_Eye"
            or coco_key == "L_Ear"
            or coco_key == "R_Ear"
        ):

            kp_h36m_32[:, H36M32_KEY[coco_key], :] = kp_coco[:, COCO_KEY[coco_key], :]

    kp_h36m_32[:, H36M32_KEY["Pelvis"], :] = (
        kp_coco[:, COCO_KEY["R_Hip"], :] + kp_coco[:, COCO_KEY["L_Hip"], :]
    ) / 2
    kp_h36m_32[:, H36M32_KEY["Head"], :] = kp_coco[:, COCO_KEY["Nose"], :]
    kp_h36m_32[:, H36M32_KEY["Thorax"], :] = (
        kp_coco[:, COCO_KEY["R_Shoulder"], :] + kp_coco[:, COCO_KEY["L_Shoulder"], :]
    ) / 2
    kp_h36m_32[:, H36M32_KEY["Nose"], :] = (
        kp_h36m_32[:, H36M32_KEY["Thorax"], :]
        + (
            kp_h36m_32[:, H36M32_KEY["Head"], :]
            - kp_h36m_32[:, H36M32_KEY["Thorax"], :]
        )
        * 0.4
    )

    kp_h36m_32[:, H36M32_KEY["Spin"], :] = (
        kp_h36m_32[:, H36M32_KEY["Thorax"], :] + kp_h36m_32[:, H36M32_KEY["Pelvis"], :]
    ) / 2

    return kp_h36m_32


def prepare_retrain_main(prefix, json_in, aid, pid, gid, width, height, retrain_pose):

    camera_ids, K, R_w2c, t_w2c = load_eldersim_camera(json_in)
    R_w2c_new, t_w2c_new = rebase_all(R_w2c, t_w2c, normalize_scaling=True)
    t_w2c_new = t_w2c_new * SCALE

    frames, x2d_coco, s2d_coco = load_cooco_poses_all(
        prefix + "/2d_joint_coco", camera_ids, aid, pid, gid
    )

    x3d_w_coco = triangulate_with_conf(
        x2d_coco, s2d_coco, K, R_w2c_new, t_w2c_new, (s2d_coco > 0)
    )

    x3d_w_h36m = convert_coco_to_h36m32(x3d_w_coco)

    canonical_name = retrain_pose

    output_2d, output_3d, metadata, output_cameras = savedata(
        canonical_name,
        R_w2c_new,
        t_w2c_new,
        x2d_coco,
        x3d_w_h36m,
        camera_ids,
        width,
        height,
    )

    out2d = prefix + "/data_2d_gafa_detectron_pt_coco.npz"
    out3d = prefix + "/data_3d_gafa.npz"
    np.savez_compressed(out2d, positions_2d=output_2d, metadata=metadata)
    np.savez_compressed(out3d, positions_3d=output_3d, cameras=output_cameras)

    print(f"Generated joint pairs for retraining: ")
    print(f"2D : {out2d}")
    print(f"3D : {out3d}")


if __name__ == "__main__":

    args = argument.parse_args()
    PREFIX = args.prefix + "/" + args.target
    AID = args.aid
    PID = args.pid
    GID = args.gid
    DATASET = args.dataset

    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())
    width = config[DATASET]["width"]
    height = config[DATASET]["height"]
    JSON_IN = args.prefix + "/results/" + args.retrain_pose + ".json"

    prepare_retrain_main(
        PREFIX, JSON_IN, AID, PID, GID, width, height, args.retrain_pose
    )
