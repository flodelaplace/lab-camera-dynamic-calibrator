import os, sys

# This script lives in pose/. Add repo root for util/argument and locate VideoPose3D.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

vp3d_path = os.path.join(_REPO_ROOT, "third_party", "VideoPose3D")
if vp3d_path not in sys.path:
    sys.path.append(vp3d_path)

import argument
from core import load_poses, op_to_coco, COCO_KEY, H36M17_KEY, select_gpu
import numpy as np
from core import OP_KEY
import matplotlib.pyplot as plt
import torch
import yaml, json
from common.loss import *
from common.model import *
from common.camera import *
from common.generators import UnchunkedGenerator
from common.visualization import render_animation


kps_left, kps_right = [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]
joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
keypoints_metadata = {
    "layout_name": "coco",
    "num_joints": 17,
    "keypoints_symmetry": [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
}


class skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 14, 15, 16]


def convert_op_to_coco(kp_op):
    kp_coco = np.full((kp_op.shape[0], 17, kp_op.shape[2]), np.nan, dtype=np.float32)
    for k_op, k_coco in op_to_coco.items():
        kp_coco[:, COCO_KEY[k_coco], :] = kp_op[:, OP_KEY[k_op], :]
    return kp_coco


def h36m_17_to_op(pose_h36m):
    J_open = 25
    T, _, N_COORD = pose_h36m.shape
    pose_op = np.full((T, J_open, N_COORD), np.nan, dtype=pose_h36m.dtype)
    for k, op_idx in OP_KEY.items():
        if k in H36M17_KEY:
            pose_op[:, op_idx, :] = pose_h36m[:, H36M17_KEY[k], :]
        pose_op[:, OP_KEY["Neck"], :] = (
            pose_op[:, OP_KEY["RShoulder"], :] + pose_op[:, OP_KEY["LShoulder"], :]
        ) / 2.0
        pose_op[:, OP_KEY["MidHip"], :] = (
            pose_op[:, OP_KEY["RHip"], :] + pose_op[:, OP_KEY["LHip"], :]
        ) / 2.0
    return pose_op


def save_json(out_dir, X3d_h36m, frames, aid, pid, gid, cid):

    X3d_op = h36m_17_to_op(X3d_h36m)
    os.makedirs(os.path.join(out_dir, "3d_joint"), exist_ok=True)

    s3d = np.logical_not(np.all(np.isnan(X3d_op), axis=2)) * 1.0
    with open(
        os.path.join(
            out_dir, "3d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"
        ),
        "w",
    ) as fp:
        data = []
        for f, s, x in zip(frames, s3d, X3d_op):

            data.append(
                {
                    "frame_index": int(f),
                    "skeleton": [{"pose": x.flatten().tolist(), "score": s.tolist()}],
                }
            )
        json.dump({"data": data}, fp, indent=2, ensure_ascii=True)


def draw_skeleton(x2d_op, x2d_coco, aid, pid, gid, out_dir, width, height):
    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)

    ax.scatter(x2d_op[:, 0], x2d_op[:, 1], label="OpenPose")
    ax.scatter(x2d_coco[:, 0], x2d_coco[:, 1], label="COCO")
    ax.legend()
    fig.savefig(
        f"{out_dir}/vis_op_to_coco_A{aid:03d}_P{pid:03d}_G{gid:03d}.png",
        bbox_inches="tight",
    )


def load_model(model, use_cuda=True):
    model_pos = TemporalModel(
        17,
        2,
        17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False,
    )

    if use_cuda and torch.cuda.is_available():
        model_pos = model_pos.cuda()

    checkpoint = torch.load(
        model, map_location=lambda storage, loc: storage
    )  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint["model_pos"])

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    return model_pos, pad, causal_shift


def evaluate(
    model_pos,
    test_generator,
    kps_left,
    kps_right,
    joints_left,
    joints_right,
    use_cuda=True,
    action=None,
    return_predictions=False,
):
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
            if use_cuda and torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[
                    1, :, joints_right + joints_left
                ]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def inference(x2d_coco, width, height, model, use_cuda=True):

    x2d_coco[..., :2] = normalize_screen_coordinates(
        x2d_coco[..., :2], w=width, h=height
    )
    input_keypoints = x2d_coco.copy()
    # model_pos, pad, causal_shift = load_model(args)

    model_pos, pad, causal_shift = load_model(model, use_cuda=use_cuda)
    gen = UnchunkedGenerator(
        None,
        None,
        [input_keypoints],
        pad=pad,
        causal_shift=causal_shift,
        augment=True,
        kps_left=kps_left,
        kps_right=kps_right,
        joints_left=joints_left,
        joints_right=joints_right,
    )
    prediction = evaluate(
        model_pos,
        gen,
        kps_left,
        kps_right,
        joints_left,
        joints_right,
        use_cuda=use_cuda,
        return_predictions=True,
    )

    return prediction


def save_animation(prediction, x2d_coco, width, height, input_video_path, viz_output):

    downsample = 20
    viz_limit = -1
    viz_size = 6
    viz_bitrate = 3000
    # viz_skip
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

    t = 0
    input_keypoints = x2d_coco.copy()
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    anim_output = {"Reconstruction": prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=width, h=height)

    render_animation(
        input_keypoints,
        keypoints_metadata,
        anim_output,
        skeleton(),
        20,
        viz_bitrate,
        np.array(70.0, dtype=np.float32),
        viz_output,
        limit=viz_limit,
        downsample=downsample,
        size=viz_size,
        input_video_path=input_video_path,
        viewport=(width, height),
        input_video_skip=1,
    )
    print("save_animation")


def inference_main(
    prefix,
    aid,
    pid,
    gid,
    cid,
    input_video_path,
    output_video_path,
    width,
    height,
    model,
    use_cuda=True,
):

    frames, x2d, s2d = load_poses(
        os.path.join(
            prefix, "2d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C{cid:03d}.json"
        )
    )
    x2d_op = x2d.reshape(len(frames), len(OP_KEY), 2)
    x2d_coco = convert_op_to_coco(x2d_op).astype("float32")
    draw_skeleton(x2d_op[0], x2d_coco[0], aid, pid, gid, prefix, width, height)
    prediction = inference(x2d_coco, width, height, model, use_cuda=use_cuda)
    save_json(prefix, prediction, frames, aid, pid, gid, cid)

    if os.path.isfile(input_video_path):
        save_animation(
            prediction, x2d_coco, width, height, input_video_path, output_video_path
        )

    return x2d_op, x2d_coco


if __name__ == "__main__":

    args = argument.parse_args()
    
    use_cuda = (args.device == "cuda")
    if use_cuda:
        try:
            select_gpu()
        except Exception as e:
            print(f"Warning: GPU selection failed ({e}). Falling back to CPU mode.")
            use_cuda = False
    
    PREFIX = args.prefix + "/" + args.target
    AID = args.aid
    PID = args.pid
    GID = args.gid

    DATASET = args.dataset

    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())
    width = config[DATASET]["width"]
    height = config[DATASET]["height"]
    camera_ids = config[DATASET]["camera_ids"]
    model = f"./model/{args.model}"

    for cid in camera_ids:
        print(f"############# - CAMID : {cid}")
        input_video_path = (
            args.prefix + f"/videos/A{AID:03d}_P{PID:03d}_G{GID:03d}_C{cid:03d}.mp4"
        )
        output_video_path = PREFIX + f"/A{AID:03d}_P{PID:03d}_G{GID:03d}_C{cid:03d}.mp4"

        inference_main(
            PREFIX,
            AID,
            PID,
            GID,
            cid,
            input_video_path,
            output_video_path,
            width,
            height,
            model,
            use_cuda=use_cuda,
        )

    # print(output_video_path)
