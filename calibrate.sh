#!/bin/bash
# =============================================================================
# calibrate.sh — Full extrinsic calibration pipeline
#
# Usage:
#   bash ./calibrate.sh <video_dir> <calib_toml> [output_dir] [device] [mode] [options]
#
# Arguments:
#   video_dir   : Folder containing the videos.
#   calib_toml  : TOML file with intrinsic parameters.
#   output_dir  : (Optional) Output folder in Axxx_Pxxx_Gxxx format.
#   device      : (Optional) 'cpu' or 'cuda'. Default: 'cuda'.
#   mode        : (Optional) 'lightweight', 'balanced', 'performance'. Default: 'balanced'.
#
# Options:
#   --height <h>      : Real height of the person in meters (e.g. 1.80) for scaling.
#   --ref_frame <f>   : Reference frame where the person is standing straight for orientation/scaling.
#   --start_frame <s> : Start frame for calibration.
#   --end_frame <e>   : End frame for calibration.
#   --frame_skip <n>  : Interval between frames used for calibration (default: 10).
#   --conf_threshold <t>: Confidence threshold for 2D keypoints (default: 0.5).
#   --save_video      : Generate and save an overlay video of 2D pose estimation.
#   --pose_engine <e> : 'rtmpose' (default) or 'metrabs'. MeTRAbs replaces RTMPose+VideoPose3D.
# =============================================================================

set -e  # Stop on error

# --- WSL2 CUDA fix ---
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

# --- Default values ---
DEVICE="cuda"
MODE="balanced"
PERSON_HEIGHT=""
REF_FRAME=""
START_FRAME=""
END_FRAME=""
FRAME_SKIP=10
CONF_THRESHOLD=0.5
SAVE_VIDEO=""
POSE_ENGINE="rtmpose"

# --- Parse arguments ---
# Positional arguments
VIDEO_DIR=$(realpath "${1}")
CALIB_TOML=$(realpath "${2}")
OUTPUT_DIR=${3:-"./data/session_$(date +%Y%m%d_%H%M%S)"}
# Shift past the main arguments to parse options
shift 3

# Parse optional named arguments
while [[ "$#" -gt 0 ]]; do
    # Nettoyage ultra-agressif : ne conserve que les lettres, chiffres, tirets et underscores
    PARAM=$(echo "$1" | tr -cd '[:alnum:]_-')
    VAL=$(echo "$2" | tr -d '\r\n')
    case $PARAM in
        --height) PERSON_HEIGHT="$VAL"; shift ;;
        --ref_frame) REF_FRAME="$VAL"; shift ;;
        --start_frame) START_FRAME="$VAL"; shift ;;
        --end_frame) END_FRAME="$VAL"; shift ;;
        --frame_skip) FRAME_SKIP="$VAL"; shift ;;
        --conf_threshold) CONF_THRESHOLD="$VAL"; shift ;;
        --save_video) SAVE_VIDEO="--save_video" ;;
        --pose_engine) POSE_ENGINE="$VAL"; shift ;;
        cuda|cpu) DEVICE="$PARAM" ;;
        lightweight|balanced|performance) MODE="$PARAM" ;;
        *) echo "Unknown parameter passed: $PARAM"; exit 1 ;;
    esac
    shift
done


OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")

# --- Extraire AID, PID, GID depuis le nom du dossier de sortie ---
BASENAME=$(basename "${OUTPUT_DIR}")
if [[ ${BASENAME} =~ A([0-9]+)_P([0-9]+)_G([0-9]+) ]]; then
    AID=$((10#${BASH_REMATCH[1]}))
    PID=$((10#${BASH_REMATCH[2]}))
    GID=$((10#${BASH_REMATCH[3]}))
else
    echo "WARNING: Le nom du dossier de sortie '${BASENAME}' ne suit pas le format Axxx_Pxxx_Gxxx. Using defaults."
    AID=1; PID=1; GID=1
fi

# Constantes internes
SUBSET="noise_1_0"
DATASET="MyDataset"
MODEL="pretrained_h36m_detectron_coco.bin"
LAMBDA1=1.
LAMBDA2=1.

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Extrinsic Camera Calibration Pipeline               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Video Dir  : ${VIDEO_DIR}"
echo "║  Calib TOML : ${CALIB_TOML}"
echo "║  Output Dir : ${OUTPUT_DIR}"
echo "║  Session IDs: AID=${AID}, PID=${PID}, GID=${GID}"
echo "║  Pose Engine: ${POSE_ENGINE}"
echo "║  Device     : ${DEVICE}         Mode: ${MODE}"
echo "║  Frame Skip : ${FRAME_SKIP}             Conf Threshold: ${CONF_THRESHOLD}"
if [ -n "$START_FRAME" ]; then
echo "║  Calib Range: Frames ${START_FRAME} to ${END_FRAME}"
fi
if [ -n "$PERSON_HEIGHT" ]; then
echo "║  Scaling    : Height=${PERSON_HEIGHT}m, Ref Frame=${REF_FRAME}"
fi
if [ -n "$SAVE_VIDEO" ]; then
echo "║  Save Video : Enabled (RTMPose overlay)"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "${OUTPUT_DIR}"

# --- Step 1: Pose extraction --------------------------------------------------
if [ "$POSE_ENGINE" = "metrabs" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[1/7] Extracting 2D+3D poses with MeTRAbs (replaces steps 1+4)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  -> frame range: ${START_FRAME:-0} to ${END_FRAME:-<end>}"

    # Check if poses already exist with matching frame range
    POSES_EXIST=false
    JOINT2D_DIR="${OUTPUT_DIR}/${SUBSET}/2d_joint"
    JOINT3D_DIR="${OUTPUT_DIR}/${SUBSET}/3d_joint"
    if [ -d "${JOINT2D_DIR}" ] && [ -d "${JOINT3D_DIR}" ]; then
        N_2D=$(find "${JOINT2D_DIR}" -name "*.json" | wc -l)
        N_3D=$(find "${JOINT3D_DIR}" -name "*.json" | wc -l)
        if [ "$N_2D" -gt 0 ] && [ "$N_2D" -eq "$N_3D" ]; then
            # Check frame range of first file
            FIRST_FILE=$(find "${JOINT2D_DIR}" -name "*.json" | sort | head -1)
            EXISTING_RANGE=$(python3 -c "
import json, sys
with open('${FIRST_FILE}') as f:
    d = json.load(f)
    print(f\"{d['data'][0]['frame_index']} {d['data'][-1]['frame_index']} {len(d['data'])}\")
" 2>/dev/null)
            if [ -n "$EXISTING_RANGE" ]; then
                EX_START=$(echo $EXISTING_RANGE | cut -d' ' -f1)
                EX_END=$(echo $EXISTING_RANGE | cut -d' ' -f2)
                EX_COUNT=$(echo $EXISTING_RANGE | cut -d' ' -f3)
                WANT_START=${START_FRAME:-0}
                WANT_END=${END_FRAME:-$EX_END}
                if [ "$EX_START" -eq "$WANT_START" ] && [ "$EX_END" -eq "$WANT_END" ]; then
                    POSES_EXIST=true
                    echo "  -> Found existing poses: ${N_2D} cameras, ${EX_COUNT} frames (${EX_START}-${EX_END})"
                    echo "  -> Skipping MeTRAbs inference (reusing cached results)"
                fi
            fi
        fi
    fi

    if [ "$POSES_EXIST" = false ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PYTHONUNBUFFERED=1 conda run --live-stream -n metrabs_opensim python -u "${SCRIPT_DIR}/metrabs_inference.py" \
            --video_dir "${VIDEO_DIR}" \
            --calib_toml "${CALIB_TOML}" \
            --output_dir "${OUTPUT_DIR}" \
            --aid ${AID} --pid ${PID} --gid ${GID} \
            --subset_name "${SUBSET}" \
            --batch_size 8 \
            $( [ -n "${START_FRAME}" ] && echo --start_frame "${START_FRAME}" ) \
            $( [ -n "${END_FRAME}" ] && echo --end_frame "${END_FRAME}" )
    fi
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[1/7] Extracting 2D poses with RTMPose..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  -> frame range: ${START_FRAME:-0} to ${END_FRAME:-<end>}"
    python3 rtmlib_inference.py --video_dir "${VIDEO_DIR}" --output_dir "${OUTPUT_DIR}" --aid ${AID} --pid ${PID} --gid ${GID} --subset_name "${SUBSET}" --device ${DEVICE} --mode ${MODE} \
        $( [ -n "${START_FRAME}" ] && echo --start_frame "${START_FRAME}" ) \
        $( [ -n "${END_FRAME}" ] && echo --end_frame "${END_FRAME}" ) \
        ${SAVE_VIDEO}
fi


## --- Step 1.5: Create frame mapping if a range is specified -------------------
#if [ -n "$START_FRAME" ] && [ -n "$END_FRAME" ]; then
#    echo ""
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    echo "[1.5] Creating frame mapping file..."
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    python3 - << PYEOF
#import json, os, sys
#import numpy as np
#output_dir = "${OUTPUT_DIR}"
#subset = "${SUBSET}"
#gid = ${GID}
#start_frame = int("${START_FRAME}")
#end_frame = int("${END_FRAME}")
#
#num_frames = end_frame - start_frame + 1
#num_joints = 25 # OpenPose format
#
#frame_indices = list(range(start_frame, end_frame + 1))
## Create a dummy 3D skeleton for compatibility with other scripts
#dummy_skeleton = np.full((num_frames, num_joints, 3), None).tolist()
#
#mapping_data = {"frame_indices": frame_indices, "skeleton": dummy_skeleton}
#
#output_path = os.path.join(output_dir, subset, f"skeleton_w_G{gid:03d}.json")
#os.makedirs(os.path.dirname(output_path), exist_ok=True)
#with open(output_path, "w") as f:
#    json.dump(mapping_data, f)
#print(f"  -> Compatible mapping file created: {output_path}")
#PYEOF
#fi


# --- Step 2: Create cameras_Gxxx.json -----------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/7] Reading intrinsics from TOML..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CAM_NAMES=()
while IFS= read -r -d $'\0' file; do BNAME=$(basename "$file"); CAM_NAMES+=("${BNAME%.*}"); done < <(find "${VIDEO_DIR}" -maxdepth 1 \( -name "*.mp4" -o -name "*.MP4" \) -print0 | sort -zV)
python3 create_cameras_from_toml.py --toml "${CALIB_TOML}" --output_dir "${OUTPUT_DIR}/${SUBSET}" --gid ${GID} --cam_names "${CAM_NAMES[@]}"


# --- Step 2.5 (Formerly 1.5): Create frame mapping -------------------
if [ -n "$START_FRAME" ] && [ -n "$END_FRAME" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[2.5] Creating frame mapping file..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 - << PYEOF
import json, os, sys
import numpy as np
output_dir = "${OUTPUT_DIR}"
subset = "${SUBSET}"
gid = ${GID}
start_frame = int("${START_FRAME}")
end_frame = int("${END_FRAME}")

num_frames = end_frame - start_frame + 1
# Detect joint count from existing 2D files (25 for RTMPose, 26 for MeTRAbs)
import glob as _g
_jf = sorted(_g.glob(os.path.join(output_dir, subset, "2d_joint", "*.json")))
if _jf:
    with open(_jf[0]) as _f:
        import json as _j
        num_joints = len(_j.load(_f)["data"][0]["skeleton"][0]["score"])
else:
    num_joints = 25

# RTMPose keeps original frame numbers (e.g. 1500 to 2499)
frame_indices = list(range(start_frame, end_frame + 1))
dummy_skeleton = np.full((num_frames, num_joints, 3), None).tolist()

mapping_data = {"frame_indices": frame_indices, "skeleton": dummy_skeleton}

output_path = os.path.join(output_dir, subset, f"skeleton_w_G{gid:03d}.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(mapping_data, f)
print(f"  -> Compatible mapping file created: {output_path}")
PYEOF
fi

# --- Step 3: Update configuration -----------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/7] Updating configuration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - << PYEOF
import yaml, json, sys, os, glob, re
script_dir = os.getcwd() # Or use a more robust path if needed
output_dir, subset, calib_toml, aid, pid, gid = "${OUTPUT_DIR}", "${SUBSET}", "${CALIB_TOML}", ${AID}, ${PID}, ${GID}
cam_file = os.path.join(output_dir, subset, f"cameras_G{gid:03d}.json")
with open(cam_file) as f: n_cams = len(json.load(f)["CAMID"])
jfiles = sorted(glob.glob(os.path.join(output_dir, subset, "2d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C*.json")))
with open(jfiles[0]) as f:
    jdata = json.load(f)
    n_frames = len(jdata["data"])
    n_joints = len(jdata["data"][0]["skeleton"][0]["score"])
with open("./config/config.yaml") as f: config = yaml.safe_load(f)
config["MyDataset"] = {"width": 1920, "height": 1080, "scale": 1, "frame_rate": 30, "camera_ids": list(range(1, n_cams + 1)), "available_joints": list(range(n_joints)), "ransac_th_2d": 200.0, "ransac_th_3d": 1.0}
config.update({"aid": aid, "pid": pid, "gid": gid})
with open("./config/config.yaml", "w") as f: yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print(f"Config updated: {n_cams} cameras, {n_frames} frames")
PYEOF

# --- Step 4: Lifting 2D -> 3D ------------------------------------------------
if [ "$POSE_ENGINE" = "metrabs" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[4/7] Skipped (3D already extracted by MeTRAbs in step 1)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[4/7] Lifting 2D -> 3D with VideoPose3D..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    bash ./inference.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${MODEL} ${DATASET} ${DEVICE}
fi

# --- Step 5: Extrinsic calibration ------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/7] Extrinsic calibration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  → Running linear calibration by chunks..."

# Data are already cropped, just calibrate on all locally available frames (index 0 to N-1)
bash ./calib_linear.sh --conf_threshold ${CONF_THRESHOLD} "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${FRAME_SKIP} ${DATASET}

# Verify linear calibration result exists
FINAL_LINEAR_JSON="${OUTPUT_DIR}/results/linear_1_0.json"
if [ ! -f "${FINAL_LINEAR_JSON}" ]; then
    echo "ERROR: Linear calibration result not found: ${FINAL_LINEAR_JSON}"
    echo "Aborting bundle adjustment. Please check calib_linear.sh output and logs."
    exit 1
fi
echo "  → Bundle Adjustment (linear)..."
bash ./ba.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${FRAME_SKIP} ${LAMBDA1} ${LAMBDA2} linear_1_0 ${DATASET} false true ${CONF_THRESHOLD}

# --- Step 6: Evaluation and Visualization ---------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/7] Evaluation and Visualization..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
declare -A MRE_SCORES
BEST_MRE=99999
BEST_CALIB=""
for CALIB in linear_1_0 linear_1_0_ba; do
    if [ -f "${OUTPUT_DIR}/results/${CALIB}.json" ]; then
        OUTPUT=$(python3 evaluate_calibration.py --prefix "${OUTPUT_DIR}" --calib "${CALIB}" --video_dir "${VIDEO_DIR}" --visualize --conf_threshold ${CONF_THRESHOLD} \
            $( [ -n "${START_FRAME}" ] && echo --start_frame "${START_FRAME}" ))
        echo "${OUTPUT}"
        MRE=$(echo "${OUTPUT}" | grep "Global MRE" | awk '{print $4}')
        if [ -n "$MRE" ]; then
            MRE_SCORES[$CALIB]=$MRE
            if (( $(echo "$MRE < $BEST_MRE" | bc -l) )); then BEST_MRE=$MRE; BEST_CALIB=$CALIB; fi
        fi
        echo "  → 3D Visualization for ${CALIB}..."
        python3 visualize_results.py --prefix "${OUTPUT_DIR}" --subset "${SUBSET}" --calib "${CALIB}" --dataset ${DATASET} --output "${OUTPUT_DIR}/results/camera/visu_3d_${CALIB}.gif" --conf_threshold ${CONF_THRESHOLD} || echo "  ⚠ Visu ${CALIB} failed"
    fi
done

# --- Step 7: Scaling and Orientation (Optional) -------------------------------
FINAL_TOML_PATH="${OUTPUT_DIR}/results/Calib_scene_calibrated.toml"
FINAL_CALIB_NAME=""
if [ -n "$PERSON_HEIGHT" ] && [ -n "$REF_FRAME" ] && [ -n "$BEST_CALIB" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[7/7] Scaling, Orientation and Final Visualization..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # --- Reference frame mapping ---
    MAPPED_REF_FRAME=$REF_FRAME
    if [ -n "$START_FRAME" ]; then
        MAPPED_REF_FRAME=$((REF_FRAME - START_FRAME))
        echo "  -> Reference frame re-mapped from ${REF_FRAME} to index ${MAPPED_REF_FRAME} to match cropped data."
    fi

    python3 scale_scene.py \
        --prefix "${OUTPUT_DIR}" \
        --calib "${BEST_CALIB}" \
        --height ${PERSON_HEIGHT} \
        --frame_idx ${MAPPED_REF_FRAME} \
        --input_toml "${CALIB_TOML}" \
        --export_toml "${FINAL_TOML_PATH}" \
        --video_dir "${VIDEO_DIR}" \
        --conf_threshold ${CONF_THRESHOLD} \
        --pose_engine ${POSE_ENGINE}

    FINAL_CALIB_NAME="${BEST_CALIB}_oriented_scaled"
    if [ -f "${OUTPUT_DIR}/results/${FINAL_CALIB_NAME}.json" ]; then
        echo "  → Final visualization..."
        python3 visualize_results.py \
            --prefix  "${OUTPUT_DIR}" \
            --subset  "${SUBSET}" \
            --calib   "${FINAL_CALIB_NAME}" \
            --dataset ${DATASET} \
            --output  "${OUTPUT_DIR}/results/camera/visu_3d_FINAL.gif" \
            --export_trc "${OUTPUT_DIR}/results/3d_skeleton_FINAL.trc" \
            --conf_threshold ${CONF_THRESHOLD}
    fi
fi

# --- Final Summary ------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      ✅  DONE !                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Results in : ${OUTPUT_DIR}/results/"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  📊 MRE Summary Table (Mean Reprojection Error)              ║"
echo "║──────────────────────────────────────────────────────────────║"
printf "║ %-25s | %-s\n" "Method" "MRE (pixels)"
echo "║──────────────────────────────────────────────────────────────║"
for CALIB in "${!MRE_SCORES[@]}"; do
    if [ "$CALIB" == "$BEST_CALIB" ]; then
        printf "║⭐ %-23s | %-s\n" "$CALIB" "${MRE_SCORES[$CALIB]}"
    else
        printf "║  %-24s | %-s\n" "$CALIB" "${MRE_SCORES[$CALIB]}"
    fi
done
if [ -n "$FINAL_CALIB_NAME" ]; then
    echo "║──────────────────────────────────────────────────────────────║"
    echo "║  Final TOML file generated:                                  ║"
    printf "║    results/%s\n" "$(basename ${FINAL_TOML_PATH})"
    echo "║  Final TRC file (3D Poses):                                  ║"
    echo "║    results/3d_skeleton_FINAL.trc                             ║"
    echo "║  Final visualization:                                        ║"
    echo "║    results/camera/visu_3d_FINAL.gif                          ║"
else
    echo "║──────────────────────────────────────────────────────────────║"
    echo "║  To generate a final TOML, rerun with options:               ║"
    echo "║    --height <h> --ref_frame <f>                              ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
