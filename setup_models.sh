#!/bin/bash

# 1. VideoPose3D
# Please prepare VideoPose3D in ./third_party/VideoPose3D/ directory.
# If it's a git repo, you can use: git submodule update --init --recursive
# Otherwise, we ensure the directory exists and try to initialize it if it's a submodule.

if [ ! -d "./third_party/VideoPose3D" ]; then
    echo "Creating third_party directory..."
    mkdir -p ./third_party
fi

if [ -f ".gitmodules" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
else
    # If not a git repo, we try to clone it manually if missing
    if [ ! -f "./third_party/VideoPose3D/common/model.py" ]; then
        echo "No .gitmodules found and VideoPose3D missing. Cloning manually..."
        rm -rf ./third_party/VideoPose3D
        git clone https://github.com/facebookresearch/VideoPose3D ./third_party/VideoPose3D
    fi
fi

# 2. Pretrained Models
mkdir -p ./model
cd ./model

echo "Downloading VideoPose3D pretrained model..."
wget -nc https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin

echo "Model setup complete!"
cd ..
