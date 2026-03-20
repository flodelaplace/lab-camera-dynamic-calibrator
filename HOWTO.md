# Usage Guide

This guide explains how to calibrate camera extrinsics from your own videos using this pipeline.

### 1. Prepare Your Data

1. Place your synchronized videos (e.g., `.mp4`, `.avi`, `.mov`) in a directory. Ensure that the cameras are fixed and the subject is moving across the capture volume.
2. Prepare a `Calib.toml` file containing your camera names and their intrinsic parameters, matching the exact format used by Pose2Sim.

*(You can use the provided `demo/` folder to test the pipeline right away. It contains 4 synchronized videos and a `Calib_scene.toml` file).*

### 2. Run the Full Calibration Pipeline

We provide an automated script `calibrate.sh` that chains all the necessary steps: 2D pose extraction, 3D lifting, linear calibration, bundle adjustment, and real-world metric scaling.

**Example using the demo dataset:**

```bash
bash ./calibrate.sh \
    "demo/" \
    "demo/Calib_scene.toml" \
    "./output/demo_calibration" \
    "cuda" \
    "balanced" \
    --height 1.80 \
    --ref_frame 5
```

#### Arguments explained:
1. `"demo/"` : Path to the folder containing your synchronized videos.
2. `"demo/Calib_scene.toml"` : Path to your initial intrinsics TOML file.
3. `"./output/demo_calibration"` : *(Optional)* Output directory.
4. `"cuda"` : *(Optional)* Compute device (`cuda` or `cpu`).
5. `"balanced"` : *(Optional)* RTMPose model size (`lightweight`, `balanced`, or `performance`).

#### Optional Flags for Scaling & Cropping:
* `--height 1.80` : The real-world height of the subject in meters. Used to scale the final extrinsics to true metric coordinates.
* `--ref_frame 5` : A specific frame where the subject is standing straight, feet on the ground. This is used to re-orient the coordinate system (Y-axis vertical, XZ-plane on the ground).
* `--start_frame` & `--end_frame` : Only process a specific segment of the video. Highly recommended if your videos are long.

### 3. Check the Results

Once the script finishes, it will generate a summary of the Mean Reprojection Error (MRE) for each optimization step. 

You can find all outputs in your output directory (`output/demo_calibration/results/`):
* **`Calib_scene_calibrated.toml`** : The final calibrated intrinsic and extrinsic parameters, ready to be used in Pose2Sim or OpenCap.
* **`camera/visu_3d_FINAL.gif`** : A 3D animation of the triangulated skeleton and the estimated camera positions.


---

### (Alternative) Manual Step-by-Step Execution

If you prefer to run the steps manually to debug or inspect intermediate files:

1. **Extract 2D Keypoints (RTMPose)**
```bash
python rtmlib_inference.py \
    --video_dir demo/ \
    --output_dir output/demo_calibration \
    --aid 1 --pid 1 --gid 1 \
    --subset_name raw_rtm \
    --device cuda
```

2. **Generate Camera JSON File**
```bash
python create_cameras_from_toml.py \
    --toml demo/Calib_scene.toml \
    --output_dir output/demo_calibration/raw_rtm \
    --gid 1 \
    --cam_names "cam1" "cam2" "cam3" "cam4" # Adjust to match your TOML sections
```

3. **Lift to 3D Keypoints (VideoPose3D)**
```bash
sh inference.sh output/demo_calibration 1 1 1 raw_rtm pretrained_h36m_detectron_coco.bin Custom
```

4. **Calibrate Camera Extrinsics**
```bash
sh calib_linear.sh output/demo_calibration 1 1 1 raw_rtm 10 Custom
sh ba.sh output/demo_calibration 1 1 1 10 1. 100000. linear_1_0 Custom false true
```

5. **Scale and Align to True World Coordinates**
```bash
python scale_scene.py \
    --prefix output/demo_calibration \
    --calib linear_1_0_ba \
    --height 1.80 \
    --frame_idx 5 \
    --subset raw_rtm \
    --input_toml demo/Calib_scene.toml \
    --export_toml output/demo_calibration/results/Calib_scene_calibrated.toml \
    --video_dir demo/
```

6. **Visualization**
```bash
python visualize_results.py \
    --prefix output/demo_calibration \
    --subset raw_rtm \
    --calib linear_1_0_ba_oriented_scaled \
    --dataset Custom \
    --output output/demo_calibration/results/camera/visu_3d.mp4
```