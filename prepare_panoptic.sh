if [ $# != 8 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET, DATASET, SRC_ORIGINAL DETECTRON_DIR
    echo [e.g.] sh ./prepare_panoptic.sh ./data/A088_P088_G088 88 88 88 noise_88_0 Panoptic ./third_party/Panoptic/161029_flute1 ./data/A088_P088_G088/poses_from_vp3d
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


mkdir -p ${PREFIX}/videos
mkdir -p ${PREFIX}/poses_from_vp3d


for cam_id in 03 12 18 22
do
echo ${cam_id}
SRC_SEQUENCE=./third_party/Panoptic/161029_flute1/hdVideos/hd_00_${cam_id}.mp4
DST_VIDEO=${PREFIX}/videos/Camera_${cam_id}.mp4
echo ${SRC_SEQUENCE}
echo ${DST_VIDEO}
ffmpeg -ss 4:25 -i ${SRC_SEQUENCE} -t 1:35 -c copy ${DST_VIDEO}
done
mv ${PREFIX}/videos/Camera_03.mp4 ${PREFIX}/videos/Camera_3.mp4


# Must be absolute path
for cam_id in 3 12 18 22
do

python ./third_party/VideoPose3D/inference/infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir ${DETECTRON_DIR} \
    --image-ext mp4 \
    ${PREFIX}/videos/Camera_${cam_id}.mp4
done
cd third_party/VideoPose3D/data/
for cam_id in 3 12 18 22
do
# This script must be launched from the "data" directory. otherwise makes an error
python prepare_data_2d_custom.py -i ${DETECTRON_DIR} -o Camera_${cam_id}.mp4
done

for cam_id in 3 12 18 22
do
mv ./data_2d_custom_Camera_${cam_id}.mp4.npz ${DETECTRON_DIR}/
done


cd ../../../
cd ./third_party/Panoptic/161029_flute1
tar -xvf hdPose3d_stage1_coco19.tar 
cd ../../../

python3 ./third_party/Panoptic/prepare_panoptic.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --src_original ${SRC_ORIGINAL}


# sh ./prepare_panoptic.sh ./data/A088_P088_G088 88 88 88 noise_88_0 Panoptic ./third_party/Panoptic/161029_flute1 /home/slee/pub/extrinsic-camera-calibration-from-a-moving-person/data/A088_P088_G088/poses_from_vp3d
