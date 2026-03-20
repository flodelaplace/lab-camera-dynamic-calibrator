#!/bin/bash
if [ $# != 7 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET, DATASET MODEL
    echo [e.g.] sh ./inference.sh ./data/A023_P102_G003 23 102 3 noise_3_0 pretrained_h36m_detectron_coco.bin SynADL
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5} 
MODEL=${6}
DATASET=${7}
echo "Inference by VideoPose3D"

python3 inference.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --model ${MODEL}




