#!/bin/bash
if [ $# != 3 ]; then
    echo
    echo $0
    echo [Usage] PREFIX, EPOCH, VIS_ID
    echo [e.g.] sh ./retrain_vis.sh ./data 100 retrain1
fi

PREFIX=${1}
EPOCH=${2}
VIS_ID=${3}


## GAFA (retrain)
sh ./vis.sh ${PREFIX}/A077_P077_G077 77 77 77 linear_77_${EPOCH} GAFA camera
sh ./vis.sh ${PREFIX}/A077_P077_G077 77 77 77 linear_77_${EPOCH}_ba GAFA camera

sh ./vis.sh ${PREFIX}/A077_P077_G077 77 77 77 noise_77_${EPOCH} GAFA ${VIS_ID}
sh ./vis.sh ${PREFIX}/A088_P088_G088 88 88 88 noise_88_${EPOCH} Panoptic ${VIS_ID}
sh ./vis.sh ${PREFIX}/A099_P099_G099 99 99 99 noise_99_${EPOCH} H36M ${VIS_ID}

