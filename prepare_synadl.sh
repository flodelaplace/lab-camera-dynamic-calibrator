#!/bin/bash
if [ $# != 5 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TARGET
    echo [e.g.] sh $0 ./third_party/SynADL 23 102 3 ./data/
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
TARGET=${5}
echo "############## CALIB SynADL ##############"

#CAM="0"   # to calibrate all the cameras from 1 to 28
CAM="21 23 25 27"
python3 calib_synadl.py --prefix ${PREFIX} --target ${TARGET} --aid ${AID} --pid ${PID} --gid ${GID} --calib $CAM


SRC_VIDEOS=${PREFIX}/RGB/
if [ -e ${SRC_VIDEOS} ]; then
    echo "############## COPY VIDEO ##############"
    AID0=$(printf "A%03d\n" "${AID}")
    PID0=$(printf "P%03d\n" "${PID}")
    GID0=$(printf "G%03d\n" "${GID}")


    DST_VIDEOS=${TARGET}/${AID0}_${PID0}_${GID0}/videos
    mkdir -p ${DST_VIDEOS}

    for cid in $CAM
    do
        CID0=$(printf "C%03d\n" "${cid}")
        cp ${SRC_VIDEOS}/${AID0}_${PID0}_${GID0}_${CID0}.avi ${DST_VIDEOS}
        filename=${DST_VIDEOS}/${AID0}_${PID0}_${GID0}_${CID0}.avi
        ffmpeg -i $filename  "${filename%.*}.mp4"
        rm ${filename}
    done
fi
echo "############## COMPLETE ##############"
