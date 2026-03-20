#!/bin/bash


if [ $# != 9 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, FRAME_SKIP, LAMBDA1, LAMBDA2, TARGET, DATASET
    echo [e.g.] sh ./run_all.sh ./data/A077_P077_G077 77 77 77 20 1. 100000. 77 GAFA
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



sh ./inference.sh ${PREFIX} ${AID} ${PID} ${GID} noise_${TARGET}_0 pretrained_h36m_detectron_coco.bin ${DATASET}

sh ./calib_linear.sh ${PREFIX} ${AID} ${PID} ${GID} noise_${TARGET}_0 ${FRAME_SKIP} ${DATASET} false 
sh ./ba.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} ${LAMBDA1} ${LAMBDA2}  linear_${TARGET}_0 ${DATASET} false true

sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} linear_${TARGET}_0 ${DATASET}
sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} linear_${TARGET}_0_ba ${DATASET}

sh ./calib_ransac.sh ${PREFIX} ${AID} ${PID} ${GID} noise_${TARGET}_0 ${FRAME_SKIP} ${DATASET}  
sh ./ba.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} ${LAMBDA1} ${LAMBDA2} ransac_${TARGET}_0 ${DATASET} false false
sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} ransac_${TARGET}_0 ${DATASET}
sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} ransac_${TARGET}_0_ba ${DATASET}


sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} noise_${TARGET}_0 ${DATASET} 2d
sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} gt_subset ${DATASET} 2d
sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} noise_${TARGET}_0 ${DATASET} 3d
sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} gt_subset ${DATASET} 3d
sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} linear_${TARGET}_0 ${DATASET} camera
sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} linear_${TARGET}_0_ba ${DATASET} camera

if [ $DATASET != "SynADL" ]; then
    sh ./calib_linear.sh ${PREFIX} ${AID} ${PID} ${GID} noise_${TARGET}_0 ${FRAME_SKIP} ${DATASET} true
    sh ./ba.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP}  ${LAMBDA1} ${LAMBDA2}  linear_${TARGET}_0 ${DATASET} true true
    sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} linear_${TARGET}_0_mask ${DATASET}
    sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} linear_${TARGET}_0_mask_ba ${DATASET}
    sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} linear_${TARGET}_0_mask ${DATASET} camera
    sh ./vis.sh ${PREFIX} ${AID} ${PID} ${GID} linear_${TARGET}_0_mask_ba ${DATASET} camera
fi


