#!/bin/bash
if [ $# != 6 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, TRIALS, NOISE_LEVEL
    echo [e.g.] sh ./add_noise.sh ./data/A023_P102_G003 23 102 3 1 3
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}  
TRIALS=${5}
NOISE_LEVEL=${6}


python3 add_noise.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --trials ${TRIALS} --noise_level ${NOISE_LEVEL}



