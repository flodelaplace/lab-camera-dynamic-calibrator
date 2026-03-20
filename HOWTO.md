# Usage Guide

This guide explains how to calibrate camera extrinsics from your own videos using this pipeline.

### 1. Prepare Your Data

1. Place your synchronized videos (e.g., `.mp4`, `.avi`, `.mov`) in a directory (e.g., `input/my_calibration/videos`). Ensure that the cameras are fixed and the subject is moving across the capture volume.
2. Prepare a `Calib.toml` file containing your camera names and their intrinsic parameters, matching the exact format used by Pose2Sim. Place it in `input/my_calibration/Calib.toml`.

### 2. Generate Camera JSON File

Convert your `Calib.toml` into the specific `.json` structure expected by this pipeline using the utility script:

```bash
python create_cameras_from_toml.py \
    --toml input/my_calibration/Calib.toml \
    --output_dir output/my_calibration/raw_rtm \
    --gid 1 \
    --cam_names "cam1" "cam2" "cam3"
```
*(The suffix `G001` is used by the pipeline to track a specific "group" or trial. The cam_names must match the section headers in your TOML).*

### 3. Extract 2D Keypoints (RTMPose)

Run RTMPose to extract 2D keypoints for the video segments you want to process.

```bash
python rtmlib_inference.py \
    --video_dir input/my_calibration/videos \
    --output_dir output/my_calibration \
    --aid 1 --pid 1 --gid 1 \
    --subset_name raw_rtm \
    --device cuda \
    --start_frame 0 \
    --end_frame 300
```
This will process the first 300 frames of every video in the directory and save the 2D joint predictions in OpenPose-25 JSON format under `output/my_calibration/raw_rtm/2d_joint`.

### 4. Lift to 3D Keypoints (VideoPose3D)

Next, we use VideoPose3D to estimate unscaled 3D poses from the 2D keypoints. *Note: You must have downloaded the VideoPose3D pretrained model into `model/` as instructed in `README.md`.*

```bash
# This script executes the VideoPose3D pipeline locally
sh inference.sh output/my_calibration 1 1 1 raw_rtm pretrained_h36m_detectron_coco.bin Custom
```

### 5. Calibrate Camera Extrinsics

With 2D and 3D joints ready, we can now run the core calibration script. This will use linear calibration followed by bundle adjustment.

```bash
sh run_custom.sh output/my_calibration 1 1 1
```
*Wait for the optimization to finish. The results will be stored in `output/my_calibration/results/`.*

### 6. Scale and Align to True World Coordinates

The outputs from the calibration step are up to an unknown scale and arbitrary reference frame. To bring the extrinsics into true world coordinates (e.g., in meters, relative to the floor), use your reference object or the person's height.

```bash
python scale_scene.py \
    --prefix output/my_calibration \
    --calib linear_1_0_ba \
    --height 1.80 \
    --frame_idx 150 \
    --subset raw_rtm \
    --input_toml input/my_calibration/Calib.toml \
    --export_toml output/my_calibration/Calib_scaled.toml \
    --video_dir input/my_calibration/videos
```
*(This assumes frame 150 has the person standing straight. It will export the final true-scale calibration back to a Pose2Sim-compatible TOML file).*

### 7. Evaluation and Visualization

You can visualize the calibrated cameras and triangulated 3D points to verify accuracy:

```bash
python visualize_results.py \
    --prefix output/my_calibration \
    --subset raw_rtm \
    --calib linear_1_0_ba_oriented_scaled \
    --dataset Custom \
    --output output/my_calibration/results/camera/visu_3d.mp4
```
