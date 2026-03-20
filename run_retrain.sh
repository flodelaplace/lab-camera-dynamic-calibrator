#!/bin/bash



echo "############## 1st retrain ##############"
sh ./retrain.sh ./data/A077_P077_G077 77 77 77 noise_77_0 GAFA linear_77_0_mask_ba 20 pretrained_h36m_detectron_coco.bin 100
echo "############## Calibration (retrain) ##############"
sh ./retrain_calib.sh ./data/A077_P077_G077 77 77 77 noise_77_100 GAFA 20 epoch_100.bin 100
# echo "############## 1st vis ##############"
# sh ./retrain_vis.sh ./data 100 retrain1
echo "############## 2st retrain ##############"
sh ./retrain.sh ./data/A077_P077_G077 77 77 77 noise_77_100 GAFA linear_77_100_ba 20 epoch_100.bin 120

echo "############## Calibration (retrain) ##############"
sh ./retrain_calib.sh ./data/A077_P077_G077 77 77 77 noise_77_120 GAFA 20 epoch_120.bin 120
# echo "############## 2st vis ##############"
# sh ./retrain_vis.sh ./data 120 retrain2


