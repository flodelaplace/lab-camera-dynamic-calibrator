#!/bin/bash

# This script downloads the required pretrained models for VideoPose3D
# RTMPose models are handled automatically by rtmlib during the first run.

mkdir -p ./model
cd ./model

echo "Downloading VideoPose3D pretrained model..."
wget -nc https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin

echo "Model setup complete!"
cd ..
