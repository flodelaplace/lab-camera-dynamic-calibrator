#!/bin/bash
if [ $# != 7 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, EID, FRAME_SKIP, TARGET, DATASET
    echo [e.g.] sh ./eval.sh ./data/A023_P102_G003 23 102 3 15 linear_3_0 SynADL
  
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
FRAME_SKIP=${5}
TARGET=${6}
DATASET=${7}

echo ${DATASET}
python3 eval.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --frame_skip ${FRAME_SKIP} --target ${TARGET} --dataset ${DATASET}



