#!/bin/bash
if [ $# != 10 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET DATASET RETRAIN_POSE MODEL UNTILL_EPOCH
    echo [e.g.] sh ./retrain.sh ./data/A077_P077_G077 77 77 77 noise_77_0 GAFA linear_77_0_mask_ba 20 pretrained_h36m_detectron_coco.bin 100

    exit
fi

PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5} 
DATASET=${6}
RETRAIN_POSE=${7}
FRAME_SKIP=${8}
MODEL=${9}
UNTILL_EPOCH=${10}

echo "############## Prepare retrain ##############"
sh ./prepare_retrain.sh ${PREFIX} ${AID} ${PID} ${GID} ${TARGET} ${DATASET} ${RETRAIN_POSE}
echo model-${MODEL}
echo epoch-${UNTILL_EPOCH}
echo "############## Retrain ##############"
python3 retrain.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --retrain_target_epoch ${UNTILL_EPOCH} --retrain_resume ${MODEL} --retrain_pose ${RETRAIN_POSE}




# echo "############## COPY (Ours) ##############"

# TARGET=noise_77_${UNTILL_EPOCH}
# mkdir -p ${PREFIX}/${TARGET} 
# cp -r ${PREFIX}/noise_77_0/2d_joint ${PREFIX}/${TARGET}/
# cp -r ${PREFIX}/noise_77_0/2d_joint_coco ${PREFIX}/${TARGET}/
# cp ${PREFIX}/noise_77_0/cameras_G077.json ${PREFIX}/${TARGET}/
# cp ${PREFIX}/noise_77_0/skeleton_w_G077.json ${PREFIX}/${TARGET}/


# echo "############## COPY (H36M) ##############"

# PREFIX_H36M=./data/A099_P099_G099
# TARGET_H36M=noise_99_${UNTILL_EPOCH}
# mkdir -p ${PREFIX_H36M}/${TARGET_H36M} 

# cp -r ${PREFIX_H36M}/noise_99_0/2d_joint ${PREFIX_H36M}/${TARGET_H36M}/
# cp ${PREFIX_H36M}/noise_99_0/cameras_G099.json ${PREFIX_H36M}/${TARGET_H36M}/
# cp ${PREFIX_H36M}/noise_99_0/skeleton_w_G099.json ${PREFIX_H36M}/${TARGET_H36M}/


# echo "############## COPY (PANOPTIC) ##############"

# PREFIX_PANOPTIC=./data/A088_P088_G088
# TARGET_PANOPTIC=noise_88_${UNTILL_EPOCH}
# mkdir -p ${PREFIX_PANOPTIC}/${TARGET_PANOPTIC} 

# cp -r ${PREFIX_PANOPTIC}/noise_88_0/2d_joint ${PREFIX_PANOPTIC}/${TARGET_PANOPTIC}/
# cp ${PREFIX_PANOPTIC}/noise_88_0/cameras_G088.json ${PREFIX_PANOPTIC}/${TARGET_PANOPTIC}/
# cp ${PREFIX_PANOPTIC}/noise_88_0/skeleton_w_G088.json ${PREFIX_PANOPTIC}/${TARGET_PANOPTIC}/

# echo "############## 3D pose Inference ##############"

# # ## Inference 
# MODEL=epoch_${UNTILL_EPOCH}.bin
# RETRAIN_POSE=linear_77_${UNTILL_EPOCH}_ba
# sh ./inference.sh ${PREFIX} ${AID} ${PID} ${GID} ${TARGET} ${MODEL} ${DATASET}
# sh ./inference.sh ./data/A099_P099_G099 99 99 99 ${TARGET_H36M} ${MODEL} H36M
# sh ./inference.sh ./data/A088_P088_G088 88 88 88 ${TARGET_PANOPTIC} ${MODEL} PANOPTIC
# echo "############## Calibration (Linear) ##############"
# sh ./calib_linear.sh ${PREFIX} ${AID} ${PID} ${GID} ${TARGET} ${FRAME_SKIP}  ${DATASET} false
# echo "############## Calibration (BA) ##############"
# sh ./ba.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} 1. 10. linear_77_${UNTILL_EPOCH} ${DATASET} false false

# echo "############## Evaluation ##############"

# sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} linear_77_${UNTILL_EPOCH} ${DATASET}
# sh ./eval.sh ${PREFIX} ${AID} ${PID} ${GID} ${FRAME_SKIP} linear_77_${UNTILL_EPOCH}_ba ${DATASET}


# echo "############## Prepare retrain ##############"
# if [ $UNTILL_EPOCH = 100 ]; then
# sh ./prepare_retrain.sh ${PREFIX} ${AID} ${PID} ${GID} ${TARGET} ${DATASET} ${RETRAIN_POSE}
# fi


# done


# ## GAFA (retrain)
# sh ./vis.sh ${PREFIX}  ${AID} ${PID} ${GID} linear_77_100 ${DATASET} camera
# sh ./vis.sh ${PREFIX}  ${AID} ${PID} ${GID} linear_77_100_ba ${DATASET} camera
# sh ./vis.sh ${PREFIX}  ${AID} ${PID} ${GID} linear_77_120 ${DATASET} camera
# sh ./vis.sh ${PREFIX}  ${AID} ${PID} ${GID} linear_77_120_ba ${DATASET} camera