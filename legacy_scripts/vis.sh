#!/bin/bash
if [ $# != 7 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET VIS_TYPE
    echo [e.g.] sh ./vis.sh ./data/A023_P102_G003 23 102 3 noise_3_0 SynADL 2d 
    echo [e.g.] sh ./vis.sh ./data/A023_P102_G003 23 102 3 gt_subset SynADL 2d 
    echo [e.g.] sh ./vis.sh ./data/A023_P102_G003 23 102 3 noise_3_0 SynADL 3d 
    echo [e.g.] sh ./vis.sh ./data/A023_P102_G003 23 102 3 noise_3_0 SynADL camera 
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5}     
DATASET=${6}
VIS_TYPE=${7}
echo ${VIS_TYPE}
python3 vis.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --dataset ${DATASET} --vis_type ${VIS_TYPE} --target ${TARGET}
