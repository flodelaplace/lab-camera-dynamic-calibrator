#!/usr/bin/env python3
"""Bundle Adjustment runner with OOM auto-retry.

Wraps ``calibration/ba.py`` in a subprocess loop that increments
``--frame_skip`` by 5 on failure (typically OOM), up to a hard cap of 60.

This file replaces the older ``scripts/ba.sh``. The positional argument
order is preserved so the calling pattern from ``calibrate.sh`` is
unchanged.

Usage:
    python scripts/run_ba.py PREFIX AID PID GID FRAME_SKIP \\
        LAMBDA1 LAMBDA2 TARGET DATASET OBS_MASK SAVE_OBS_MASK [CONF_THRESHOLD]

Example:
    python scripts/run_ba.py ./data/A023_P102_G003 23 102 3 1 \\
        1. 100000. linear_3_0 SynADL false false
"""
import os
import subprocess
import sys

MAX_FRAME_SKIP = 60
FRAME_SKIP_STEP = 5

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BA_SCRIPT = os.path.join(REPO_ROOT, "calibration", "ba.py")

USAGE = (
    "[Usage] PREFIX AID PID GID FRAME_SKIP LAMBDA1 LAMBDA2 TARGET DATASET "
    "OBS_MASK SAVE_OBS_MASK [CONF_THRESHOLD]\n"
    "[e.g.]  python scripts/run_ba.py ./data/A023_P102_G003 23 102 3 1 "
    "1. 100000. linear_3_0 SynADL false false"
)


def main(argv):
    if len(argv) < 11:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    prefix, aid, pid, gid, frame_skip = argv[0:5]
    lambda1, lambda2, target, dataset = argv[5:9]
    obs_mask, save_obs_mask = argv[9:11]
    conf_threshold = argv[11] if len(argv) > 11 else "0.5"

    current_frame_skip = int(frame_skip)
    while True:
        print(f"Attempting Bundle Adjustment with FRAME_SKIP={current_frame_skip}...")
        cmd = [
            sys.executable, BA_SCRIPT,
            "--prefix", prefix,
            "--aid", aid,
            "--pid", pid,
            "--gid", gid,
            "--frame_skip", str(current_frame_skip),
            "--ba_lambda1", lambda1,
            "--ba_lambda2", lambda2,
            "--target", target,
            "--dataset", dataset,
            "--obs_mask", obs_mask,
            "--th_obs_mask", "20",
            "--save_obs_mask", save_obs_mask,
            "--conf_threshold", conf_threshold,
        ]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"Bundle Adjustment successful with FRAME_SKIP={current_frame_skip}.")
            return

        print(
            f"WARN: Bundle Adjustment failed (exit {result.returncode}). "
            "This might be due to an out-of-memory error."
        )
        prev_frame_skip = current_frame_skip
        current_frame_skip += FRAME_SKIP_STEP
        if current_frame_skip > MAX_FRAME_SKIP:
            print(
                f"ERROR: Bundle Adjustment failed even with FRAME_SKIP up to "
                f"{prev_frame_skip}. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Retrying with a larger frame skip: {current_frame_skip}...\n")


if __name__ == "__main__":
    main(sys.argv[1:])
