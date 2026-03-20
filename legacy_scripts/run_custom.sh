#!/bin/bash
# =============================================================================
# run_custom.sh — Full pipeline with your own videos using RTMPose (BodyWithFeet)
#
# Usage:
#   sh ./run_custom.sh <video_dir> <output_dir> <aid> <pid> <gid> <dataset_name> [device] [mode]
#
# Example:
#   sh ./run_custom.sh ./my_videos ./data/A001_P001_G001 1 1 1 MyDataset cuda balanced
#
# Arguments:
#   video_dir     : folder with your camera video files (e.g. cam1.mp4, cam2.mp4, ...)
#   output_dir    : working directory where data will be stored
#   aid           : action ID (any number, e.g. 1)
#   pid           : person ID (any number, e.g. 1)
#   gid           : group/scene ID (any number, e.g. 1)
#   dataset_name  : name of your dataset entry in config/config.yaml (e.g. MyDataset)
#   device        : (optional) cpu or cuda  [default: cuda]
#   mode          : (optional) lightweight / balanced / performance  [default: balanced]
# =============================================================================

if [ $# -lt 6 ]; then
    echo ""
    echo "Usage: sh $0 <video_dir> <output_dir> <aid> <pid> <gid> <dataset_name> [device] [mode]"
    echo "e.g. : sh $0 ./my_videos ./data/A001_P001_G001 1 1 1 MyDataset cuda balanced"
    exit 1
fi

VIDEO_DIR=${1}
OUTPUT_DIR=${2}
AID=${3}
PID=${4}
GID=${5}
DATASET=${6}
DEVICE=${7:-cuda}
MODE=${8:-balanced}

SUBSET="noise_1_0"
MODEL="pretrained_h36m_detectron_coco.bin"
FRAME_SKIP=1
LAMBDA1=1.
LAMBDA2=100000.

echo "============================================================"
echo " run_custom.sh"
echo "  video_dir   : ${VIDEO_DIR}"
echo "  output_dir  : ${OUTPUT_DIR}"
echo "  AID/PID/GID : ${AID}/${PID}/${GID}"
echo "  dataset     : ${DATASET}"
echo "  device      : ${DEVICE}"
echo "  mode        : ${MODE}"
echo "============================================================"

# ------------------------------------------------------------
# Step 1 — Extract 2D poses with RTMPose (Halpe26 -> OpenPose25)
# ------------------------------------------------------------
echo ""
echo "[1/4] Extracting 2D poses with RTMPose (BodyWithFeet)..."
python3 rtmlib_inference.py \
    --video_dir  "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --aid ${AID} --pid ${PID} --gid ${GID} \
    --subset_name "${SUBSET}" \
    --device  ${DEVICE} \
    --mode    ${MODE}

if [ $? -ne 0 ]; then
    echo "ERROR: rtmlib_inference.py failed."
    exit 1
fi

# ------------------------------------------------------------
# Step 2 — Lift 2D -> 3D with VideoPose3D
# ------------------------------------------------------------
echo ""
echo "[2/4] Lifting 2D -> 3D with VideoPose3D..."
sh ./inference.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${MODEL} ${DATASET}

if [ $? -ne 0 ]; then
    echo "ERROR: inference.sh failed."
    exit 1
fi

# ------------------------------------------------------------
# Step 2b — Create cameras_G{gid}.json and skeleton_w placeholder
#           (intrinsic K required by calib_linear + ba)
# ------------------------------------------------------------
echo ""
echo "[2b/4] Creating cameras and skeleton placeholder files..."

# Read width/height from config
WIDTH=$(python3 -c "import yaml; c=yaml.safe_load(open('./config/config.yaml')); print(c['${DATASET}']['width'])")
HEIGHT=$(python3 -c "import yaml; c=yaml.safe_load(open('./config/config.yaml')); print(c['${DATASET}']['height'])")
CAM_IDS=$(python3 -c "import yaml; c=yaml.safe_load(open('./config/config.yaml')); print(' '.join(str(x) for x in c['${DATASET}']['camera_ids']))")

python3 create_cameras_json.py \
    --output_dir "${OUTPUT_DIR}/${SUBSET}" \
    --gid ${GID} \
    --camera_ids ${CAM_IDS} \
    --width ${WIDTH} \
    --height ${HEIGHT}

if [ $? -ne 0 ]; then
    echo "ERROR: create_cameras_json.py failed."
    exit 1
fi

# ------------------------------------------------------------
# Step 3 — Extrinsic calibration (linear + bundle adjustment)
# ------------------------------------------------------------
echo ""
echo "[3/4] Running linear calibration..."
sh ./calib_linear.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${FRAME_SKIP} ${DATASET} false

if [ $? -ne 0 ]; then
    echo "ERROR: calib_linear.sh failed."
    exit 1
fi

echo ""
echo "[3/4] Running bundle adjustment (linear)..."
sh ./ba.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${FRAME_SKIP} ${LAMBDA1} ${LAMBDA2} linear_1_0 ${DATASET} false true

if [ $? -ne 0 ]; then
    echo "ERROR: ba.sh (linear) failed."
    exit 1
fi

# ------------------------------------------------------------
# Step 4 — RANSAC calibration + bundle adjustment
# ------------------------------------------------------------
echo ""
echo "[4/4] Running RANSAC calibration..."
sh ./calib_ransac.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${FRAME_SKIP} ${DATASET}

if [ $? -ne 0 ]; then
    echo "ERROR: calib_ransac.sh failed."
    exit 1
fi

echo ""
echo "[4/4] Running bundle adjustment (RANSAC)..."
sh ./ba.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${FRAME_SKIP} ${LAMBDA1} ${LAMBDA2} ransac_1_0 ${DATASET} false false

if [ $? -ne 0 ]; then
    echo "ERROR: ba.sh (ransac) failed."
    exit 1
fi

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Done! Extrinsic calibration results:"
echo ""
echo "  Linear:           ${OUTPUT_DIR}/results/linear_1_0.json"
echo "  Linear + BA:      ${OUTPUT_DIR}/results/linear_1_0_ba.json"
echo "  RANSAC:           ${OUTPUT_DIR}/results/ransac_1_0.json"
echo "  RANSAC + BA:      ${OUTPUT_DIR}/results/ransac_1_0_ba.json"
echo ""
echo "Each JSON contains R (rotation 3x3) and t (translation 3x1)"
echo "for each camera = the extrinsic parameters you need."
echo "============================================================"

