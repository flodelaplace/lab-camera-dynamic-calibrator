#!/bin/bash
if [ $# != 7 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, EID, TARGET, FRAME_SKIP
    echo [e.g.] sh ./calib_ransac.sh ./data/A023_P102_G003 23 102 3 noise_3_0 15 SynADL
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5}     
FRAME_SKIP=${6}
DATASET=${7}


python3 calib_ransac.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --frame_skip ${FRAME_SKIP} --dataset ${DATASET}



