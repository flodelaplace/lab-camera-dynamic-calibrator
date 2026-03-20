"""
rtmlib_inference.py
-------------------
Extract 2D poses from multi-camera videos using RTMPose (BodyWithFeet / Halpe26)
and save them in the OpenPose-25 JSON format expected by this repo.

Usage:
    python rtmlib_inference.py \
        --video_dir  ./my_videos \
        --output_dir ./data/A001_P001_G001 \
        --aid 1 --pid 1 --gid 1 \
        --width 1920 --height 1080 \
        --device cuda \
        --mode balanced

The script expects video files in --video_dir named like:
    cam1.mp4, cam2.mp4, ...   (any name is fine, they get sorted)

It will produce:
    <output_dir>/raw_rtm/           <- 2D poses in JSON (OpenPose-25 format)
        A001_P001_G001_C001.json
        A001_P001_G001_C002.json
        ...

After this script, run:
    sh ./inference.sh <output_dir> 1 1 1 raw_rtm pretrained_h36m_detectron_coco.bin MyDataset
then:
    sh ./run_all.sh <output_dir> 1 1 1 <frame_skip> 1. 100000. 1 MyDataset
"""

import argparse
import json
import os
import sys
import glob

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Halpe26 keypoint indices (rtmlib BodyWithFeet output)
# ---------------------------------------------------------------------------
HALPE26_KEY = {
    "Nose":       0,
    "LEye":       1,
    "REye":       2,
    "LEar":       3,
    "REar":       4,
    "LShoulder":  5,
    "RShoulder":  6,
    "LElbow":     7,
    "RElbow":     8,
    "LWrist":     9,
    "RWrist":     10,
    "LHip":       11,
    "RHip":       12,
    "LKnee":      13,
    "RKnee":      14,
    "LAnkle":     15,
    "RAnkle":     16,
    "Head":       17,   # top of head
    "Neck":       18,
    "MidHip":     19,
    "LBigToe":    20,
    "RBigToe":    21,
    "LSmallToe":  22,
    "RSmallToe":  23,
    "LHeel":      24,
    "RHeel":      25,
}

# ---------------------------------------------------------------------------
# OpenPose-25 keypoint indices (format expected by this repo)
# ---------------------------------------------------------------------------
OP_KEY = {
    "Nose":      0,
    "Neck":      1,
    "RShoulder": 2,
    "RElbow":    3,
    "RWrist":    4,
    "LShoulder": 5,
    "LElbow":    6,
    "LWrist":    7,
    "MidHip":    8,
    "RHip":      9,
    "RKnee":     10,
    "RAnkle":    11,
    "LHip":      12,
    "LKnee":     13,
    "LAnkle":    14,
    "REye":      15,
    "LEye":      16,
    "REar":      17,
    "LEar":      18,
    "LBigToe":   19,
    "LSmallToe": 20,
    "LHeel":     21,
    "RBigToe":   22,
    "RSmallToe": 23,
    "RHeel":     24,
}

# ---------------------------------------------------------------------------
# Halpe26 -> OpenPose25 mapping
# All joints are direct, Neck and MidHip exist directly in Halpe26.
# ---------------------------------------------------------------------------
HALPE26_TO_OP25 = {op_name: HALPE26_KEY[op_name] for op_name in OP_KEY if op_name in HALPE26_KEY}
# Every OP25 joint has a direct counterpart in Halpe26 — no interpolation needed.


def get_best_person(keypoints_all, scores_all, bboxes_all):
    """Selects the best person from multiple detections based on bounding box area."""
    if keypoints_all.ndim == 3 and len(keypoints_all) > 1:
        areas = (bboxes_all[:, 2] - bboxes_all[:, 0]) * (bboxes_all[:, 3] - bboxes_all[:, 1])
        best_idx = int(np.argmax(areas))
        return keypoints_all[best_idx], scores_all[best_idx]
    elif keypoints_all.ndim == 3:
        return keypoints_all[0], scores_all[0]
    return keypoints_all, scores_all

def halpe26_to_op25(kp_halpe, sc_halpe):
    """Converts a single person's Halpe26 skeleton to OpenPose-25."""
    kp_op = np.zeros((25, 2), dtype=np.float32)
    score_op = np.zeros(25, dtype=np.float32)
    for op_name, op_idx in OP_KEY.items():
        if op_name in HALPE26_TO_OP25:
            h_idx = HALPE26_TO_OP25[op_name]
            kp_op[op_idx] = kp_halpe[h_idx]
            score_op[op_idx] = sc_halpe[h_idx]
    return kp_op, score_op


def process_video(video_path: str, body_model, output_op25_json: str, output_halpe26_json: str, start_frame: int = None, end_frame: int = None):
    """Run RTMPose on a selected range of frames and save a 2d_joint JSON."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {os.path.basename(video_path)}  ({width}x{height}, {total_frames} frames)")
    op25_data = []
    halpe26_data = []
    # Détermine la plage de frames à traiter
    sf = start_frame if start_frame is not None else 0
    ef = end_frame if end_frame is not None else total_frames - 1
    if sf > ef or sf < 0 or ef >= total_frames:
        print(f"ERROR: Invalid frame range ({sf} to {ef}) for video {video_path}")
        cap.release()
        return width, height
    n_frames = ef - sf + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    with tqdm(total=n_frames, desc=f"  Processing frames {sf}-{ef}") as pbar:
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = sf + i
            model_output = body_model(frame)
            if len(model_output) == 3:
                keypoints_all, scores_all, bboxes_all = model_output
            else:
                keypoints_all, scores_all = model_output
                bboxes_all = None
            kp_halpe, sc_halpe = np.zeros((26, 2)), np.zeros(26)
            if keypoints_all is not None and len(keypoints_all) > 0:
                if bboxes_all is None:
                    if keypoints_all.ndim == 3:
                        bboxes_all = []
                        for j, person_kps in enumerate(keypoints_all):
                            vis_kps = person_kps[scores_all[j] > 0]
                            if len(vis_kps) > 0:
                                x1, y1 = vis_kps.min(axis=0)
                                x2, y2 = vis_kps.max(axis=0)
                                bboxes_all.append([x1, y1, x2, y2])
                            else:
                                bboxes_all.append([0,0,0,0])
                        bboxes_all = np.array(bboxes_all)
                    else:
                        vis_kps = keypoints_all[scores_all > 0]
                        if len(vis_kps) > 0:
                            x1, y1 = vis_kps.min(axis=0)
                            x2, y2 = vis_kps.max(axis=0)
                            bboxes_all = np.array([x1, y1, x2, y2])
                        else:
                            bboxes_all = np.array([0,0,0,0])
                kp_halpe, sc_halpe = get_best_person(keypoints_all, scores_all, bboxes_all)
            kp_op25, sc_op25 = halpe26_to_op25(kp_halpe, sc_halpe)
            op25_data.append({"frame_index": frame_idx, "skeleton": [{"pose": kp_op25.flatten().tolist(), "score": sc_op25.tolist()}]})
            halpe26_data.append({"frame_index": frame_idx, "skeleton": [{"pose": kp_halpe.flatten().tolist(), "score": sc_halpe.tolist()}]})
            pbar.update(1)
    cap.release()
    with open(output_op25_json, "w") as f:
        json.dump({"data": op25_data}, f, indent=2, ensure_ascii=True)
    print(f"  Saved {len(op25_data)} OpenPose-25 frames -> {output_op25_json}")
    with open(output_halpe26_json, "w") as f:
        json.dump({"data": halpe26_data}, f, indent=2, ensure_ascii=True)
    print(f"  Saved {len(halpe26_data)} Halpe26 frames -> {output_halpe26_json}")

    return width, height


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2D poses with RTMPose (BodyWithFeet/Halpe26) "
                    "and save in OpenPose-25 JSON format for the calibration repo."
    )
    parser.add_argument("--video_dir",  required=True,
                        help="Folder containing the camera video files (mp4/avi/...)")
    parser.add_argument("--output_dir", required=True,
                        help="Output prefix, e.g. ./data/A001_P001_G001")
    parser.add_argument("--aid", type=int, default=1, help="Action ID (default 1)")
    parser.add_argument("--pid", type=int, default=1, help="Person ID (default 1)")
    parser.add_argument("--gid", type=int, default=1, help="Group/Scene ID (default 1)")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--backend", default="onnxruntime",
                        choices=["onnxruntime", "opencv"],
                        help="ONNX backend (default: onnxruntime)")
    parser.add_argument("--mode", default="balanced",
                        choices=["lightweight", "balanced", "performance"],
                        help="RTMPose model size (default: balanced)")
    parser.add_argument("--subset_name", default="noise_1_0",
                        help="Name of the subset folder inside output_dir (default: noise_1_0)")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Start frame index for pose extraction (default: 0)")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="End frame index for pose extraction (default: last frame)")
    args = parser.parse_args()

    # ---- collect & sort video files ----------------------------------------
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.video_dir, ext)))
    video_files = sorted(video_files)

    if len(video_files) == 0:
        print(f"ERROR: No video files found in {args.video_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s):")
    for i, v in enumerate(video_files):
        print(f"  [{i+1}] {os.path.basename(v)}")

    # ---- load RTMPose model -------------------------------------------------
    try:
        from rtmlib import BodyWithFeet
    except ImportError:
        print("ERROR: rtmlib is not installed. Run: pip install rtmlib")
        sys.exit(1)

    print(f"\nLoading BodyWithFeet model (mode={args.mode}, device={args.device}) ...")
    body_model = BodyWithFeet(
        mode=args.mode,
        to_openpose=False,   # keep Halpe26 output — we do the conversion ourselves
        backend=args.backend,
        device=args.device,
    )

    # ---- process each video -------------------------------------------------
    out_op25_dir = os.path.join(args.output_dir, args.subset_name, "2d_joint")
    out_halpe26_dir = os.path.join(args.output_dir, args.subset_name, "2d_joint_halpe26")
    os.makedirs(out_op25_dir, exist_ok=True)
    os.makedirs(out_halpe26_dir, exist_ok=True)

    widths, heights = [], []
    for cam_idx, video_path in enumerate(video_files, start=1):
        cid = cam_idx
        base_name = f"A{args.aid:03d}_P{args.pid:03d}_G{args.gid:03d}_C{cid:03d}.json"
        output_op25_json = os.path.join(out_op25_dir, base_name)
        output_halpe26_json = os.path.join(out_halpe26_dir, base_name)
        
        if os.path.exists(output_op25_json) and os.path.exists(output_halpe26_json):
            print(f"\n[Camera {cid}] Skipping, output files already exist.")
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cap.release()
            continue

        print(f"\n[Camera {cid}]")
        w, h = process_video(
            video_path, body_model, output_op25_json, output_halpe26_json,
            start_frame=args.start_frame, end_frame=args.end_frame
        )
        widths.append(w)
        heights.append(h)

    # ---- print next-step instructions ---------------------------------------
    if not widths or not heights:
        print("\nNo videos were processed. Exiting.")
        sys.exit(0)
        
    width  = widths[0]
    height = heights[0]
    n_cams = len(video_files)
    cam_ids_str = ", ".join(str(i+1) for i in range(n_cams))

    print("\n" + "="*60)
    print("2D pose extraction complete!")
    print(f"Output: {out_op25_dir}")
    print(f"Also saved raw Halpe26 data to: {out_halpe26_dir}")
    print(f"Cameras: {cam_ids_str}  |  Resolution: {width}x{height}")
    print("="*60)


if __name__ == "__main__":
    main()
