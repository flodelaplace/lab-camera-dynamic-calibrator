#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ $# -lt 7 ] || [ $# -gt 8 ]; then
    echo
    echo $0
    echo "[Usage] PREFIX AID, PID, GID, TARGET, DATASET MODEL [DEVICE]"
    echo "[e.g.] bash scripts/inference.sh ./data/A023_P102_G003 23 102 3 noise_3_0 pretrained_h36m_detectron_coco.bin SynADL cuda"
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5}
MODEL=${6}
DATASET=${7}
DEVICE=${8:-cuda}
echo "Inference by VideoPose3D (Device: ${DEVICE})"

python3 "${REPO_ROOT}/pose/inference.py" --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --model ${MODEL} --device ${DEVICE}




