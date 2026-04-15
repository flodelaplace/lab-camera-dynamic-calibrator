if [ $# != 8 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET, DATASET, SRC_ORIGINAL DETECTRON_DIR 
    echo [e.g.] sh ./prepare_gafa.sh ./data/A077_P077_G077 77 77 77 noise_77_0 GAFA ./third_party/GAFA/lab/1013_2  ./data/A077_P077_G077/poses_from_vp3d
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
for cam_id in 1_2 2_2 5_2 6_2 7_2
do
echo ${cam_id}
SRC_SEQUENCE=./third_party/GAFA/lab/1013_2/Camera_${cam_id}
DST_VIDEO=${PREFIX}/videos/Camera_${cam_id}.mp4
echo ${SRC_SEQUENCE}
ffmpeg -r 30 -start_number 130 -i ${SRC_SEQUENCE}/%06d.jpg -vcodec libx264 ${DST_VIDEO}
done


# Must be absolute path
for cam_id in 1_2 2_2 5_2 6_2 7_2
do

python ./third_party/VideoPose3D/inference/infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir ${DETECTRON_DIR} \
    --image-ext mp4 \
    ${PREFIX}/videos/Camera_${cam_id}.mp4
done
cd third_party/VideoPose3D/data/
for cam_id in 1_2 2_2 5_2 6_2 7_2
do
# This script must be launched from the "data" directory. otherwise makes an error
python prepare_data_2d_custom.py -i ${DETECTRON_DIR} -o Camera_${cam_id}.mp4
done

for cam_id in 1_2 2_2 5_2 6_2 7_2
do
mv ./data_2d_custom_Camera_${cam_id}.mp4.npz ${DETECTRON_DIR}/
done

cd ../../../
python3 ./third_party/GAFA/prepare_GAFA.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --target ${TARGET} --dataset ${DATASET} --src_original ${SRC_ORIGINAL}
