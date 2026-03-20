# Dynamic Extrinsic Camera Calibrator

This repository is an improved and customized version of [Extrinsic Camera Calibration From a Moving Person](https://github.com/kyotovision-public/extrinsic-camera-calibration-from-a-moving-person) (IROS 2022 and RA-L). 

While the core optimization engine for calculating extrinsic parameters is preserved, several major improvements and modernizations have been integrated into this pipeline:

## ✨ Key Improvements & Features
* **Modern 2D Pose Estimation**: Replaced Detectron with **RTMPose** (via `rtmlib`) for faster and more accurate 2D joint detection.
* **3D Pose Lifting**: We continue to use **VideoPose3D** for 2D-to-3D pose lifting.
* **Streamlined Optimization**: Focuses on **Linear** and **Linear Bundle Adjustment** for the extrinsic parameter calculation (RANSAC is available in the codebase but removed from the main pipeline).
* **True World Coordinates**: Extrinsic parameters are now computed in a **true world coordinate system with real dimensions** by providing a reference frame at the end of the pipeline.
* **Standardized Intrinsics Input**: Intrinsic parameters are now provided via a `Calib.toml` file, identical to the format used in [Pose2Sim](https://github.com/perfanalytics/pose2sim).
* **Partial Video Processing**: Support for selecting and processing only a specific segment/part of a video instead of the whole file.

## 1. Prerequisites

We recommend using Anaconda/Miniconda to manage the environment. The required packages and their specific versions (including PyTorch with CUDA support and RTMPose dependencies) are listed in `conda_linux.yaml`.

```bash
conda env create -f conda_linux.yaml
conda activate human_calib
```

## 2. Models & Third-party Dependencies

After setting up the conda environment, you need to download the pretrained models for VideoPose3D. We provide a quick setup script for this:

```bash
bash setup_models.sh
```
*(Note: RTMPose models are automatically downloaded and handled by `rtmlib` upon first execution).*

## 3. Fast Demo Setup 🚀

Want to test the pipeline right away? A demo dataset with 4 synchronized videos and a base `Calib_scene.toml` is provided in the `demo/` folder.

You can run the full calibration on this demo dataset in one single command:

```bash
bash ./calibrate.sh \
    "demo/videos" \
    "demo/Calib_scene.toml" \
    "./output/demo_calibration" \
    "cuda" \
    "balanced" \
    --height 1.80 \
    --ref_frame 5
```

Once finished, the final fully-scaled extrinsic calibration TOML and a 3D visualization GIF will be available in `./output/demo_calibration/results/`.

## 4. Full Guide & Pipeline Execution

To use this pipeline with **your own data**, or to understand the inner workings step-by-step (2D extraction -> 3D lifting -> Calibration -> World Scaling), please check out the **[HOWTO.md](HOWTO.md)**.

## Acknowledgments & Citations

If you use this code, please acknowledge the original authors of the extrinsic calibration engine:
```bibtex
@ARTICLE{9834083,
  author={Lee, Sang-Eun and Shibata, Keisuke and Nonaka, Soma and Nobuhara, Shohei and Nishino, Ko},
  journal={IEEE Robotics and Automation Letters},
  title={Extrinsic Camera Calibration From a Moving Person},
  year={2022},
  volume={7},
  number={4},
  pages={10344--10351},
  doi={10.1109/LRA.2022.3192629}}
```