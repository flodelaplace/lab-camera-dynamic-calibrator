if [ $# != 8 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET, DATASET, SRC_ORIGINAL DETECTRON_DIR
    echo [e.g.] sh ./prepare_h36m.sh ./data/A099_P099_G099 99 99 99 noise_99_0 H36M ./third_party/H36M ./data/A099_P099_G099/poses_from_vp3d
    exit
fi



PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}  
TARGET=${5}
DATASET=${6}
SRC_ORIGINAL=${7}
DETECTRON_DIR=$(realpath ${8})

mkdir -p ${PREFIX}/videos
mkdir -p ${PREFIX}/poses_from_vp3d


cp ./third_party/H36M/S11/Videos/"Walking 1.54138969.mp4" ${PREFIX}/videos
cp ./third_party/H36M/S11/Videos/"Walking 1.55011271.mp4" ${PREFIX}/videos
cp ./third_party/H36M/S11/Videos/"Walking 1.58860488.mp4" ${PREFIX}/videos
cp ./third_party/H36M/S11/Videos/"Walking 1.60457274.mp4" ${PREFIX}/videos

mv ${PREFIX}/videos/"Walking 1.54138969.mp4" ${PREFIX}/videos/Camera_1.mp4
mv ${PREFIX}/videos/"Walking 1.55011271.mp4" ${PREFIX}/videos/Camera_2.mp4
mv ${PREFIX}/videos/"Walking 1.58860488.mp4" ${PREFIX}/videos/Camera_3.mp4
mv ${PREFIX}/videos/"Walking 1.60457274.mp4" ${PREFIX}/videos/Camera_4.mp4


# Must be absolute path
for cam_id in 1 2 3 4
do

python ./third_party/VideoPose3D/inference/infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir ${DETECTRON_DIR} \
    --image-ext mp4 \
    ${PREFIX}/videos/Camera_${cam_id}.mp4
done
cd third_party/VideoPose3D/data/
for cam_id in 1 2 3 4
do
# This script must be launched from the "data" directory. otherwise makes an error
python prepare_data_2d_custom.py -i ${DETECTRON_DIR} -o Camera_${cam_id}.mp4
done

for cam_id in 1 2 3 4
do
mv ./data_2d_custom_Camera_${cam_id}.mp4.npz ${DETECTRON_DIR}/
done
cd ../../../
python3 ./third_party/H36M/prepare_h36m.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --src_original ${SRC_ORIGINAL}

# sh ./prepare_h36m.sh ./data/A099_P099_G099 99 99 99 noise_99_0 H36M ./third_party/H36M /home/slee/pub/extrinsic-camera-calibration-from-a-moving-person/data/A099_P099_G099/poses_from_vp3d
