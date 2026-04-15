#!/bin/bash
if [ $# != 7 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET RETRAIN_POSE
    echo [e.g.] sh ./prepare_retrain.sh ./data/A077_P077_G077 77 77 77 noise_77_0 GAFA linear_77_0_ba
    exit
fi

PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5} 
DATASET=${6}
RETRAIN_POSE=${7}


python3 prepare_retrain.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --retrain_pose ${RETRAIN_POSE}
