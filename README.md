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

## 3. Input Data Format

Instead of relying on the original datasets' rigid formats, this repository is designed to be more flexible:
1. **Intrinsics**: Provide your camera intrinsic parameters in a `Calib.toml` file (Pose2Sim format). You can generate the initial camera setup using `create_cameras_from_toml.py`.
2. **Videos**: Place your synchronized videos in your designated input folder. You can configure the pipeline to process only a specific time window.

## 4. Pipeline Execution overview

Check out [HOWTO.md](HOWTO.md) for a step-by-step example of running the calibration.

1. **2D Pose Extraction**: Run the RTMPose inference script (`rtmlib_inference.py`) to extract 2D keypoints from the videos.
2. **3D Lifting**: Run VideoPose3D inference to lift the 2D keypoints to 3D.
3. **Calibration**: Run the linear calibration and bundle adjustment (`calib_linear.py` / `ba.py`) to estimate the extrinsic parameters.
4. **World Scaling**: Apply your reference frame to scale and align the extrinsic parameters into the true real-world coordinate system.
5. **Evaluation / Visualization**: Use `visualize_results.py` and `evaluate_calibration.py` to verify the accuracy of the calibration.

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