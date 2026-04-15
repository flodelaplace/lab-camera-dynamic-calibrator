#!/usr/bin/env python3
"""Chunked linear calibration runner.

Replaces ``scripts/calib_linear.sh``. Splits the requested frame range into
1000-frame chunks, runs ``calibration/calib_linear.py`` on each, evaluates
each chunk via ``postprocessing/evaluate_calibration.py``, and copies the
chunk with the lowest MRE to the final result file.

Optional named flags (--start_frame, --end_frame, --conf_threshold) may
appear before or after the positional arguments.

Usage:
    python scripts/run_calib_linear.py [--start_frame S] [--end_frame E] \\
        [--conf_threshold T] PREFIX AID PID GID TARGET FRAME_SKIP DATASET

Example:
    python scripts/run_calib_linear.py ./data/A023_P102_G003 23 102 3 \\
        noise_3_0 1 SynADL
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

CHUNK_SIZE = 1000  # process 1000 frames at a time (visibility filter selects best within)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_SCRIPT = os.path.join(REPO_ROOT, "calibration", "calib_linear.py")
EVAL_SCRIPT = os.path.join(REPO_ROOT, "postprocessing", "evaluate_calibration.py")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start_frame", type=int, default=None)
    p.add_argument("--end_frame", type=int, default=None)
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("prefix")
    p.add_argument("aid", type=int)
    p.add_argument("pid", type=int)
    p.add_argument("gid", type=int)
    p.add_argument("target")
    p.add_argument("frame_skip", type=int)
    p.add_argument("dataset")
    return p.parse_args(argv)


def find_min_frames(json_dir):
    """Return the smallest 'data' length across all *.json files in json_dir."""
    files = glob.glob(os.path.join(json_dir, "*.json"))
    if not files:
        return None
    min_n = None
    for f in files:
        with open(f) as fp:
            n = len(json.load(fp)["data"])
        if min_n is None or n < min_n:
            min_n = n
    return min_n


def map_video_frames_to_indices(skeleton_file, req_start, req_end):
    """Map original video frame numbers to JSON indices via 'frame_indices'.

    Returns (rs, re) JSON indices, or (None, None) if mapping fails.
    rs is the first index whose frame_index >= req_start (or None if req_start is None).
    re is the last index whose frame_index <= req_end (or None if req_end is None).
    """
    with open(skeleton_file) as fp:
        s = json.load(fp)
    fi = s.get("frame_indices")
    if fi is None:
        return None, None
    rs = re_idx = None
    try:
        if req_start is not None:
            rs = next((i for i, v in enumerate(fi) if int(v) >= req_start), -1)
        if req_end is not None:
            j = next((i for i, v in enumerate(fi) if int(v) > req_end), None)
            re_idx = (len(fi) - 1) if j is None else (j - 1)
    except Exception:
        return None, None
    return rs, re_idx


def run_chunk(args, frame_start, frame_end, chunk_id, total_chunks):
    print(f"\n--- Processing Chunk {chunk_id}/{total_chunks} (Frames {frame_start}-{frame_end}) ---")
    cmd = [
        sys.executable, CALIB_SCRIPT,
        "--prefix", args.prefix,
        "--aid", str(args.aid),
        "--pid", str(args.pid),
        "--gid", str(args.gid),
        "--target", args.target,
        "--frame_skip", str(args.frame_skip),
        "--dataset", args.dataset,
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_end),
        "--chunk_id", str(chunk_id),
        "--conf_threshold", str(args.conf_threshold),
    ]
    subprocess.run(cmd, check=False)


def evaluate_chunk(args, chunk_id):
    """Run evaluate_calibration on one chunk and return its Global MRE (or None)."""
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--prefix", args.prefix,
        "--calib", f"chunks/linear_chunk_{chunk_id}",
        "--conf_threshold", str(args.conf_threshold),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    for line in result.stdout.splitlines():
        if "Global MRE" in line:
            parts = line.split()
            # Expect "  -> Global MRE: 12.345 pixels" — MRE is the 4th token
            try:
                return float(parts[3])
            except (IndexError, ValueError):
                return None
    return None


def derive_final_name(target):
    """Mimic bash ``linear_${TARGET#*_}``: strip everything up to & including first '_'."""
    if "_" in target:
        suffix = target.split("_", 1)[1]
    else:
        suffix = target
    return f"linear_{suffix}"


def main(argv):
    args = parse_args(argv)

    json_dir = os.path.join(args.prefix, args.target, "2d_joint")
    results_dir = os.path.join(args.prefix, "results")
    chunk_results_dir = os.path.join(results_dir, "chunks")
    os.makedirs(chunk_results_dir, exist_ok=True)

    # 1. Determine MIN_FRAMES across cameras
    print("Finding the minimum number of frames across all cameras...")
    min_frames = find_min_frames(json_dir)
    if min_frames is None:
        print(f"ERROR: no JSON files found in {json_dir}", file=sys.stderr)
        sys.exit(1)

    # 2. Map video-frame numbers → JSON indices if needed
    req_start = args.start_frame
    req_end = args.end_frame
    if req_start is not None or req_end is not None:
        looks_like_video = (
            (req_start is not None and req_start >= min_frames)
            or (req_end is not None and req_end >= min_frames)
        )
        if looks_like_video:
            skeleton_file = os.path.join(
                args.prefix, args.target, f"skeleton_w_G{args.gid:03d}.json"
            )
            if os.path.isfile(skeleton_file):
                print(f"Mapping requested video-frame numbers to JSON indices using {skeleton_file}...")
                rs, re_idx = map_video_frames_to_indices(skeleton_file, req_start, req_end)
                if rs is not None and re_idx is not None and rs >= 0 and re_idx >= 0:
                    print(f"Mapped start -> {rs}, end -> {re_idx} (JSON indices)")
                    req_start = rs
                    req_end = re_idx
                else:
                    print("WARNING: could not map requested video-frame numbers to JSON indices. Falling back to index clamping.")
            else:
                print(f"WARNING: skeleton file {skeleton_file} not found; cannot map video-frame numbers to JSON indices.")

    # 3. Resolve and clamp range
    if req_start is not None and req_end is not None:
        calib_start, calib_end = req_start, req_end
    elif req_start is not None:
        calib_start, calib_end = req_start, min_frames - 1
    elif req_end is not None:
        calib_start, calib_end = 0, req_end
    else:
        calib_start, calib_end = 0, min_frames - 1

    if calib_start < 0:
        print(f"WARNING: start frame {calib_start} < 0. Clamping to 0.")
        calib_start = 0
    max_idx = min_frames - 1
    if calib_end > max_idx:
        print(f"WARNING: end frame {calib_end} > available frames ({max_idx}). Clamping to {max_idx}.")
        calib_end = max_idx

    total_frames = calib_end - calib_start + 1
    if total_frames <= 0:
        print(
            f"ERROR: Invalid frame range after clamping ({calib_start} to {calib_end}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Processing frames from {calib_start} to {calib_end} ({total_frames} total) "
        f"in chunks of {CHUNK_SIZE}..."
    )

    # 4. Run calibration per chunk
    n_chunks = (total_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(n_chunks):
        f_start = calib_start + i * CHUNK_SIZE
        f_end = min(f_start + CHUNK_SIZE - 1, calib_end)
        run_chunk(args, f_start, f_end, i, n_chunks)

    # 5. Evaluate each chunk and find the best (lowest MRE)
    print("\n--- Evaluating all chunks to find the best calibration ---")
    chunk_files = glob.glob(os.path.join(chunk_results_dir, "linear_chunk_*.json"))

    def chunk_id_of(path):
        m = re.search(r"linear_chunk_(\d+)\.json$", path)
        return int(m.group(1)) if m else -1

    chunk_files = sorted(chunk_files, key=chunk_id_of)

    best_mre = None
    best_chunk_id = None
    best_chunk_file = None
    for chunk_file in chunk_files:
        cid = chunk_id_of(chunk_file)
        if cid < 0:
            continue
        mre = evaluate_chunk(args, cid)
        if mre is None:
            continue
        if best_mre is None or mre < best_mre:
            best_mre = mre
            best_chunk_id = cid
            best_chunk_file = chunk_file

    # 6. Copy the best chunk's output to the final result
    if best_chunk_id is None:
        print(
            "ERROR: Could not determine the best calibration chunk. No final file was created.",
            file=sys.stderr,
        )
        sys.exit(1)

    final_name = derive_final_name(args.target)  # ex: linear_1_0
    final_file = os.path.join(results_dir, f"{final_name}.json")
    print("\n--- Best result found ---")
    print(f"  -> Chunk ID: {best_chunk_id}")
    print(f"  -> MRE: {best_mre} pixels")
    print(f"  -> Copying {best_chunk_file} to {final_file}")
    shutil.copy(best_chunk_file, final_file)


if __name__ == "__main__":
    main(sys.argv[1:])
