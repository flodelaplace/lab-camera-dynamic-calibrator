#!/bin/bash
if [ $# != 11 ]; then
    echo
    echo $0
    echo "[Usage] PREFIX AID PID GID FRAME_SKIP LAMBDA1 LAMBDA2 TARGET DATASET OBS_MASK SAVE_OBS_MASK"
    echo "[e.g.] sh ./ba.sh ./data/A023_P102_G003 23 102 3 1 1. 100000. linear_3_0 SynADL false false"
    exit
fi

PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
FRAME_SKIP=${5}
LAMBDA1=${6}
LAMBDA2=${7}
TARGET=${8}
DATASET=${9}
OBS_MASK=${10}
SAVE_OBS_MASK=${11}

MAX_FRAME_SKIP=60
CURRENT_FRAME_SKIP=${FRAME_SKIP}

while true; do
    echo "Attempting Bundle Adjustment with FRAME_SKIP=${CURRENT_FRAME_SKIP}..."

    # Run the python command
    python3 ba.py \
        --prefix ${PREFIX} \
        --aid ${AID} \
        --pid ${PID} \
        --gid ${GID} \
        --frame_skip ${CURRENT_FRAME_SKIP} \
        --ba_lambda1 ${LAMBDA1} \
        --ba_lambda2 ${LAMBDA2} \
        --target ${TARGET} \
        --dataset ${DATASET} \
        --obs_mask ${OBS_MASK} \
        --th_obs_mask 20 \
        --save_obs_mask ${SAVE_OBS_MASK}

    # Check the exit code of the last command
    if [ $? -eq 0 ]; then
        echo "Bundle Adjustment successful with FRAME_SKIP=${CURRENT_FRAME_SKIP}."
        break # Exit loop on success
    else
        echo "WARN: Bundle Adjustment failed (Exit Code: $?). This might be due to an out-of-memory error."

        # Increase the frame skip
        PREV_FRAME_SKIP=${CURRENT_FRAME_SKIP}
        CURRENT_FRAME_SKIP=$((CURRENT_FRAME_SKIP + 5))

        if [ ${CURRENT_FRAME_SKIP} -gt ${MAX_FRAME_SKIP} ]; then
            echo "ERROR: Bundle Adjustment failed even with FRAME_SKIP up to ${PREV_FRAME_SKIP}. Aborting."
            exit 1
        fi

        echo "Retrying with a larger frame skip: ${CURRENT_FRAME_SKIP}..."
        echo "" # Empty line for readability
    fi
done
