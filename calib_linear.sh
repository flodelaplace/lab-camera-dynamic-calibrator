#!/bin/bash
# Default values
START_FRAME=""
END_FRAME=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start_frame) START_FRAME="$2"; shift ;;
        --end_frame) END_FRAME="$2"; shift ;;
        *) break ;; # Stop parsing on the first positional argument
    esac
    shift
done

# Positional arguments
if [ $# != 7 ]; then
    echo
    echo "Usage: $0 PREFIX AID PID GID TARGET FRAME_SKIP DATASET [--start_frame S] [--end_frame E]"
    echo "[e.g.] sh ./calib_linear.sh ./data/A023_P102_G003 23 102 3 noise_3_0 1 SynADL"
    exit 1
fi

PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5}
FRAME_SKIP=${6}
DATASET=${7}

CHUNK_SIZE=500 # Process 500 frames at a time
JSON_DIR="${PREFIX}/${TARGET}/2d_joint"
RESULTS_DIR="${PREFIX}/results"
CHUNK_RESULTS_DIR="${RESULTS_DIR}/chunks"

mkdir -p "${CHUNK_RESULTS_DIR}"

# --- 1. Déterminer le nombre de frames à traiter ---
echo "Finding the minimum number of frames across all cameras..."
MIN_FRAMES=999999
while IFS= read -r -d $'\0' json_file; do
    FRAME_COUNT=$(python3 -c "import json; print(len(json.load(open('$json_file'))['data']))")
    if (( FRAME_COUNT < MIN_FRAMES )); then
        MIN_FRAMES=$FRAME_COUNT
    fi
done < <(find "${JSON_DIR}" -name "*.json" -print0)

# Utiliser l'intervalle spécifié, sinon utiliser toutes les frames
REQUEST_START=${START_FRAME:-}
REQUEST_END=${END_FRAME:-}

# If requested start/end look like video-frame numbers (i.e. larger than the
# number of entries in the JSON), try to map them to JSON indices using the
# skeleton file that contains "frame_indices" (original video frame numbers).
# This avoids errors when RTM already filtered frames and produced JSONs with
# indices 0..N-1 while the user passed original video frame numbers.
if [ -n "${REQUEST_START}" ] || [ -n "${REQUEST_END}" ]; then
    IS_VIDEO_IDX=0
    if [ -n "${REQUEST_START}" ] 2>/dev/null && [ ${REQUEST_START} -ge ${MIN_FRAMES} ] 2>/dev/null; then
        IS_VIDEO_IDX=1
    fi
    if [ -n "${REQUEST_END}" ] 2>/dev/null && [ ${REQUEST_END} -ge ${MIN_FRAMES} ] 2>/dev/null; then
        IS_VIDEO_IDX=1
    fi

    if [ ${IS_VIDEO_IDX} -eq 1 ]; then
        GID_PADDED=$(printf "%03d" "${GID}")
        SKELETON_FILE="${PREFIX}/${TARGET}/skeleton_w_G${GID_PADDED}.json"
        if [ -f "${SKELETON_FILE}" ]; then
            echo "Mapping requested video-frame numbers to JSON indices using ${SKELETON_FILE}..."
            # python prints two integers: start_index end_index (or -1 -1 on failure)
            MAP_OUT=$(python3 - <<PY

import json
s=json.load(open('${SKELETON_FILE}'))
fi=s.get('frame_indices', None)
if fi is None:
    print('-1 -1')
else:
    rs = None
    re = None
    try:
        if '${REQUEST_START}'!='':
            rs_val=int(${REQUEST_START})
            rs = next((i for i,v in enumerate(fi) if int(v) >= rs_val), -1)
        if '${REQUEST_END}'!='':
            re_val=int(${REQUEST_END})
            # last index where frame_index <= re_val
            re_idx = next((i for i,v in enumerate(fi) if int(v) > re_val), None)
            if re_idx is None:
                re = len(fi)-1
            else:
                re = re_idx-1
    except Exception:
        rs = -1
        re = -1
    if rs is None: rs = -1
    if re is None: re = -1
    print(f"{rs} {re}")
PY
)
            S_IDX=$(echo "${MAP_OUT}" | awk '{print $1}')
            E_IDX=$(echo "${MAP_OUT}" | awk '{print $2}')
            if [ "${S_IDX}" -ge 0 ] 2>/dev/null && [ "${E_IDX}" -ge 0 ] 2>/dev/null; then
                echo "Mapped start -> ${S_IDX}, end -> ${E_IDX} (JSON indices)"
                REQUEST_START=${S_IDX}
                REQUEST_END=${E_IDX}
            else
                echo "WARNING: could not map requested video-frame numbers to JSON indices. Falling back to index clamping."
            fi
        else
            echo "WARNING: skeleton file ${SKELETON_FILE} not found; cannot map video-frame numbers to JSON indices."
        fi
    fi
fi

# Clamp requested range to available frames (backward-compatible behavior)
if [ -n "${REQUEST_START}" ] && [ -n "${REQUEST_END}" ]; then
    # both provided
    CALIB_START=${REQUEST_START}
    CALIB_END=${REQUEST_END}
elif [ -n "${REQUEST_START}" ]; then
    CALIB_START=${REQUEST_START}
    CALIB_END=$((MIN_FRAMES - 1))
elif [ -n "${REQUEST_END}" ]; then
    CALIB_START=0
    CALIB_END=${REQUEST_END}
else
    CALIB_START=0
    CALIB_END=$((MIN_FRAMES - 1))
fi

# Now clamp to [0, MIN_FRAMES-1]
if [ ${CALIB_START} -lt 0 ]; then
    echo "WARNING: start frame ${CALIB_START} < 0. Clamping to 0."
    CALIB_START=0
fi
if [ ${CALIB_END} -gt $((MIN_FRAMES - 1)) ]; then
    MAX_IDX=$((MIN_FRAMES - 1))
    echo "WARNING: end frame ${CALIB_END} > available frames (${MAX_IDX}). Clamping to ${MAX_IDX}."
    CALIB_END=${MAX_IDX}
fi

TOTAL_FRAMES=$((CALIB_END - CALIB_START + 1))
if [ $TOTAL_FRAMES -le 0 ]; then
    echo "ERROR: Invalid frame range after clamping (${CALIB_START} to ${CALIB_END})."
    exit 1
fi
echo "Processing frames from ${CALIB_START} to ${CALIB_END} (${TOTAL_FRAMES} total) in chunks of ${CHUNK_SIZE}..."

# --- 2. Lancer la calibration pour chaque chunk ---
NUM_CHUNKS=$(( (TOTAL_FRAMES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    FRAME_START=$((CALIB_START + i * CHUNK_SIZE))
    FRAME_END=$((FRAME_START + CHUNK_SIZE - 1))
    if [ $FRAME_END -gt $CALIB_END ]; then
        FRAME_END=$CALIB_END
    fi

    echo ""
    echo "--- Processing Chunk ${i}/${NUM_CHUNKS} (Frames ${FRAME_START}-${FRAME_END}) ---"

    python3 calib_linear.py \
        --prefix "${PREFIX}" \
        --aid ${AID} --pid ${PID} --gid ${GID} \
        --target ${TARGET} \
        --frame_skip ${FRAME_SKIP} \
        --dataset ${DATASET} \
        --frame_start ${FRAME_START} \
        --frame_end ${FRAME_END} \
        --chunk_id ${i}
done

# --- 3. Évaluer chaque chunk et trouver le meilleur ---
echo ""
echo "--- Evaluating all chunks to find the best calibration ---"
BEST_MRE=99999
BEST_CHUNK_ID=-1
BEST_CHUNK_FILE=""

for chunk_file in $(ls -v "${CHUNK_RESULTS_DIR}"/linear_chunk_*.json 2>/dev/null); do
    CHUNK_ID=$(echo "$chunk_file" | sed -n 's/.*linear_chunk_\([0-9]*\).json/\1/p')

    OUTPUT=$(python3 evaluate_calibration.py \
        --prefix "${PREFIX}" \
        --calib "chunks/linear_chunk_${CHUNK_ID}")

    echo "${OUTPUT}"

    MRE=$(echo "${OUTPUT}" | grep "Global MRE" | awk '{print $4}')

    if [ -n "$MRE" ]; then
        if (( $(echo "$MRE < $BEST_MRE" | bc -l) )); then
            BEST_MRE=$MRE
            BEST_CHUNK_ID=$CHUNK_ID
            BEST_CHUNK_FILE=$chunk_file
        fi
    fi
done

# --- 4. Copier le meilleur résultat ---
if [ ${BEST_CHUNK_ID} -ne -1 ]; then
    FINAL_OUTPUT_NAME="linear_${TARGET#*_}" # ex: linear_1_0
    FINAL_OUTPUT_FILE="${RESULTS_DIR}/${FINAL_OUTPUT_NAME}.json"

    echo ""
    echo "--- Best result found ---"
    echo "  -> Chunk ID: ${BEST_CHUNK_ID}"
    echo "  -> MRE: ${BEST_MRE} pixels"
    echo "  -> Copying ${BEST_CHUNK_FILE} to ${FINAL_OUTPUT_FILE}"

    cp "${BEST_CHUNK_FILE}" "${FINAL_OUTPUT_FILE}"
else
    echo "ERROR: Could not determine the best calibration chunk. No final file was created."
    exit 1
fi