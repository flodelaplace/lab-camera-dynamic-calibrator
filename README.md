# Dynamic Extrinsic Camera Calibrator

A complete pipeline for **extrinsic camera calibration from a moving person**. It leverages human pose estimation to calibrate multi-camera setups without needing a checkerboard or specialized calibration patterns — just a person walking in the scene.

![Overview camera extrinsics](img/Graphical_abstract_lab_camera_dynamic_calibrator.png)

---

## Pipeline Overview

The pipeline processes synchronized multi-camera videos through a 7-step pipeline orchestrated by `calibrate.sh`:

| Step | Script | Description |
|------|--------|-------------|
| **1. Pose Extraction** | `metrabs_inference.py` or `rtmlib_inference.py` | Detect 2D keypoints (+ direct 3D with MeTRAbs) in all camera views |
| **2. Intrinsics Loading** | `create_cameras_from_toml.py` | Parse camera matrices & distortion from a Pose2Sim-compatible TOML |
| **3. Configuration** | *(inline in calibrate.sh)* | Auto-detect number of cameras, joints, frame count; write `config.yaml` |
| **4. 3D Lifting** | `inference.sh` / `inference.py` | Lift 2D→3D with VideoPose3D *(skipped when using MeTRAbs)* |
| **5. Calibration** | `calib_linear.sh` / `calib_linear.py` → `ba.sh` / `ba.py` | Linear calibration (with Procrustes init for MeTRAbs) + Bundle Adjustment |
| **6. Evaluation** | `evaluate_calibration.py` | Compute Mean Reprojection Error (MRE) per camera + visualizations |
| **7. Scaling** | `scale_scene.py` | Orient scene (gravity-aligned) and scale to metric units using person height |

**Final output:** `Calib_scene_calibrated.toml` with extrinsic parameters (R, t) for each camera in a real-world metric coordinate system.

---

## Pose Engines

Two pose estimation backends are supported:

### MeTRAbs (recommended)

[MeTRAbs](https://github.com/isarandi/metrabs) is a metric-scale 3D human pose estimator. We use the **`bml_movi_87`** skeleton (87 joints from the [BML MoVi](https://www.biomotionlab.ca/movi/) dataset), from which we extract a **26-joint calibration subset**:

- **20 virtual joint centers** (head, thorax, pelvis, hips, shoulders, elbows, wrists, hands, knees, ankles, feet)
- **2 anatomical landmarks** (backneck, sternum)
- **4 foot markers** (left/right heel, left/right toe)

These 26 joints are connected by **27 bones** covering the full body including a 4-segment spine and articulated feet — significantly richer than OpenPose's 12 bones. The mapping from `bml_movi_87` to our 26-joint format is defined in `util.py` (`METRABS_BML87_INDICES`).

MeTRAbs provides **direct metric 3D** predictions per camera, enabling **Procrustes alignment** as initialization for the linear calibration. This gives a much better starting point than purely 2D-based methods.

| Property | Value |
|----------|-------|
| Model | `metrabs_l` (EffNetV2-L backbone) via TensorFlow Hub |
| Input skeleton | `bml_movi_87` (87 joints) |
| Output skeleton | 26-joint calibration format (27 bones) |
| 3D output | Metric (millimeters), per-camera coordinate frame |
| Conda env | `metrabs` (Python 3.10, TensorFlow 2.x) |
| Speed | ~2 min/camera on GPU |

### RTMPose + VideoPose3D

The classic path uses [RTMPose](https://github.com/Tau-J/rtmlib) (via `rtmlib`) for 2D detection in **Halpe26** format, then [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) for temporal 2D→3D lifting. The 3D skeleton uses **25 OpenPose joints** with **12 bones**.

| Property | Value |
|----------|-------|
| 2D model | RTMPose (ONNX, via `rtmlib`) |
| 3D model | VideoPose3D (`pretrained_h36m_detectron_coco.bin`) |
| Output skeleton | 25 OpenPose joints (12 bones) |
| 3D output | Relative scale (not metric) |
| Conda env | `human_calib` (Python 3.8, PyTorch 1.13) |

### Comparison on demo dataset (4 cameras, 100 frames)

| | MeTRAbs | RTMPose + VP3D |
|---|---|---|
| Linear calibration MRE | 7.8 px | 152.2 px |
| After Bundle Adjustment | **3.5 px** | **8.5 px** |
| Procrustes init | Yes | No |
| Scale factor | 0.001 (metric 3D in mm) | 30.9 (arbitrary units) |

---

## 1. Installation

### Prerequisites

- **OS:** Linux or WSL2 (Windows Subsystem for Linux)
- **GPU:** NVIDIA GPU with CUDA support (recommended)
- **Conda:** Anaconda or Miniconda

### Clone the repository

```bash
git clone https://github.com/flodelaplace/lab-camera-dynamic-calibrator.git
cd lab-camera-dynamic-calibrator
```

### Main environment (`human_calib`)

This environment is **always required** — it runs the calibration pipeline, Bundle Adjustment, evaluation, and visualization. It also supports the RTMPose path.

```bash
conda env create -f conda_linux.yaml
conda activate human_calib
```

Key packages: Python 3.8, PyTorch 1.13 (CUDA 11.7), scipy, opencv, rtmlib, numba, pycalib-simple.

### MeTRAbs environment (optional, recommended)

MeTRAbs requires a **separate** conda environment (Python 3.10, TensorFlow) because it is incompatible with the PyTorch-based `human_calib` environment. The pipeline handles the environment switching automatically via `conda run`.

```bash
# Clone the MeTRAbs fork (dev branch)
cd ..
git clone -b dev https://github.com/flodelaplace/metrabs.git
cd metrabs

# Create the environment and install
conda env create -f environment.yml
conda activate metrabs
pip install -e .

cd ../lab-camera-dynamic-calibrator
```

> The MeTRAbs model (`metrabs_l`) is downloaded automatically from TensorFlow Hub on first run (~1.5 GB). No manual download needed.

See the [MeTRAbs fork repository](https://github.com/flodelaplace/metrabs/tree/dev) for detailed installation and troubleshooting.

### VideoPose3D setup (RTMPose path only)

Only needed if you use the RTMPose + VideoPose3D pipeline:

```bash
conda activate human_calib
bash setup_models.sh
```

This clones VideoPose3D into `./third_party/VideoPose3D` and downloads the pretrained weights into `./model/`.

### WSL2 GPU fix

If you encounter `libcuda.so not found` on WSL2, this is handled automatically inside `calibrate.sh`. For manual runs:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

---

## 2. Quick Demo

A demo dataset (4 cameras, 100 frames) is included in `demo/`.

### With MeTRAbs (recommended)

```bash
conda activate human_calib

bash ./calibrate.sh \
    demo \
    demo/Calib_scene.toml \
    output/demo_metrabs \
    --pose_engine metrabs \
    --height 1.78 \
    --ref_frame 5
```

### With RTMPose + VideoPose3D

```bash
conda activate human_calib

bash ./calibrate.sh \
    demo \
    demo/Calib_scene.toml \
    output/demo_rtmpose \
    --height 1.78 \
    --ref_frame 5
```

### Output

Results are saved in `output/demo_*/results/`:

| File | Description |
|------|-------------|
| `Calib_scene_calibrated.toml` | Final calibration file (metric, gravity-aligned) |
| `camera/visu_3d_FINAL.gif` | 3D skeleton + camera positions animation |
| `3d_skeleton_FINAL.trc` | 3D poses in TRC format (for OpenSim / Mokka) |
| `MRE_visualizations/` | Per-camera best/worst reprojection error images |
| `ba_cost_live_iter*.png` | Bundle Adjustment convergence curves |

---

## 3. Usage with Your Own Data

### Prepare your input

Create a folder in `input/` with:

1. **Synchronized MP4 videos** — one per camera. File names (without extension) are used as camera identifiers and must match the TOML sections.
2. **`Calib_scene.toml`** — intrinsic parameters for each camera, in [Pose2Sim](https://github.com/perfanalytics/pose2sim) format:

```toml
[my_camera_01]
name = "my_camera_01"
size = [1920.0, 1080.0]
matrix = [[1057.46, 0.0, 942.23], [0.0, 1056.83, 535.6], [0.0, 0.0, 1.0]]
distortions = [-0.041, 0.0086, -0.0002, 0.0002]
fisheye = false
```

> **Important:** Intrinsic calibration quality is critical. Bad focal lengths or distortion coefficients are the #1 cause of poor results. Normal distortion values: k1, k2 in the range [-2, 2]. Values above 5 are suspicious.

### Run the calibration

```bash
bash ./calibrate.sh <video_dir> <calib_toml> <output_dir> [options]
```

### Required arguments

| Argument | Description |
|----------|-------------|
| `video_dir` | Folder containing synchronized MP4 videos |
| `calib_toml` | Path to the TOML file with intrinsic parameters |
| `output_dir` | Where to save results |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pose_engine <engine>` | `rtmpose` | Pose backend: `metrabs` or `rtmpose` |
| `--height <meters>` | *(none)* | Person height in meters (enables step 7: scaling + orientation) |
| `--ref_frame <n>` | *(none)* | Video frame number where person stands straight (for scaling) |
| `--start_frame <n>` | `0` | First video frame to process |
| `--end_frame <n>` | last | Last video frame to process |
| `--frame_skip <n>` | `10` | Frame subsampling interval for Bundle Adjustment |
| `--conf_threshold <t>` | `0.5` | Minimum keypoint confidence (lower = more data, more noise) |
| `--save_video` | off | Save 2D pose overlay video (RTMPose only) |
| `cuda` / `cpu` | `cuda` | Inference device |
| `lightweight` / `balanced` / `performance` | `balanced` | RTMPose model size |

### Full example

```bash
bash ./calibrate.sh \
    input/my_session \
    input/my_session/Calib_scene.toml \
    output/my_session \
    --pose_engine metrabs \
    --start_frame 200 --end_frame 2199 \
    --height 1.84 --ref_frame 2100 \
    --conf_threshold 0.5
```

This processes frames 200–2199 from all videos, calibrates using MeTRAbs, then scales the scene using the person's height (1.84m) measured at frame 2100.

### Caching

When using MeTRAbs, pose extraction results are cached in the output directory. If you re-run with the same frame range, inference is skipped automatically. To force re-extraction (e.g., after changing intrinsics), delete the cached poses:

```bash
rm -rf output/my_session/noise_1_0/2d_joint output/my_session/noise_1_0/3d_joint
```

---

## 4. Technical Details

### Calibration Pipeline

**Linear calibration** (`calib_linear.py`) computes initial extrinsic parameters:
- With MeTRAbs: **Procrustes alignment** between per-camera 3D skeletons gives R, t directly
- With RTMPose: uses bone orientation collinearity constraints (original method from the paper)
- Data is processed in chunks of 1000 frames, the chunk with lowest MRE is selected
- **Visibility filter**: only frames where >= 2/3 of cameras see the person are used

**Bundle Adjustment** (`ba.py`) refines the extrinsics by jointly optimizing:
1. **NLL** — weighted 2D reprojection error (main objective)
2. **var3d** — bone direction consistency across cameras
3. **varbone** — bone length variance across frames (regularizer)

Key BA features:
- **Auto-balanced lambda2**: the bone regularization weight is automatically computed at each iteration so that the bone term contributes ~10% of the NLL term. This works correctly regardless of the pose engine.
- **Jacobian sparsity**: a sparse Jacobian structure is provided to `scipy.least_squares`, giving 50–200x speedup on Jacobian computation.
- **Live convergence plot**: a PNG is saved every 10s showing the cost reduction curve.
- **2-pass optimization**: after the first pass, frames with reprojection error > 2x median are removed as outliers, then a second pass runs on the cleaned data.
- **Convergence**: uses `ftol=xtol=gtol=1e-8` with `max_nfev=20000` (~2 min cap with sparsity).

### MeTRAbs Quality Filtering

The MeTRAbs inference applies several quality filters before saving keypoints:

| Filter | Threshold | Effect |
|--------|-----------|--------|
| Dark/black frames | mean brightness < 15 | Detection set to None (conf=0) |
| Small bounding box | area < 0.5% of image | Rejected as false positive |
| Collapsed skeleton | 2D spread < 20px | Rejected (all joints in same spot) |
| Out-of-bounds joints | < 10px from image edge | 2D confidence reduced to × 0.1 |

The 3D confidence (`s3d`) uses the bounding box confidence only (not affected by OOB penalty), since MeTRAbs predicts full-body 3D even when 2D joints are clipped at the image edge.

### Joint and Bone Definitions

**MeTRAbs calib-26** (26 joints, 27 bones):

```
              head (0)
               |
           backneck (1)
               |
            thor (2) ── sternum (3)
           / | \
      lsho(6) |  rsho(7)         Shoulder width: lsho ─── rsho
        |     |     |
     lelb(8) pelv(4) relb(9)
        |   / | \    |
    lwri(10) | mhip(5) rwri(11)
        |  lhip(14) rhip(15)  |   Hip width: lhip ─── rhip
    lhan(12) |       | rhan(13)
          lkne(16)  rkne(17)
            |        |
          lank(18)  rank(19)
          / |        | \
    lhee(22) lfoo(20) rfoo(21) rhee(23)
       |                        |
    ltoe(24)                 rtoe(25)
```

The 26 joints are extracted from `bml_movi_87` using indices defined in `METRABS_BML87_INDICES` (see `util.py` line 195).

**RTMPose / OpenPose-25** (25 joints, 12 bones):
Standard OpenPose body-25 format with joints: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar, LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel.

### Scaling and Orientation

Step 7 (`scale_scene.py`) transforms the calibrated scene into a metric, gravity-aligned coordinate system:

1. **Ground plane**: fitted from 6 foot keypoints (heels, toes, feet centers)
2. **Vertical axis (Y)**: defined by head-to-feet vector (Y points down in OpenCV convention)
3. **Horizontal axis (X)**: defined by left-heel → right-heel direction
4. **Origin**: center of heels at ground level
5. **Scale**: computed from `measured_skeleton_height / real_person_height`

With MeTRAbs, the scaling uses the calib-26 joints directly (head=0, heels=22/23, toes=24/25, feet=20/21). With RTMPose, it uses the Halpe26 format.

---

## 5. Project Structure

```
lab-camera-dynamic-calibrator/
├── calibrate.sh              # Main pipeline orchestrator (entry point)
├── metrabs_inference.py      # MeTRAbs pose extraction (bml_movi_87 → calib-26)
├── rtmlib_inference.py       # RTMPose 2D pose detection
├── inference.py / .sh        # VideoPose3D 3D lifting
├── calib_linear.py / .sh     # Linear calibration + Procrustes init
├── ba.py / .sh               # Bundle Adjustment with Jacobian sparsity
├── scale_scene.py            # Metric scaling and gravity alignment
├── evaluate_calibration.py   # MRE evaluation and visualization
├── visualize_results.py      # 3D GIF rendering (auto-detects skeleton format)
├── util.py                   # Joint/bone definitions, triangulation, projection
├── argument.py               # CLI argument parsing
├── create_cameras_from_toml.py  # TOML → cameras JSON converter
├── my_dataset.py             # Dataset class for pose data
├── config/config.yaml        # Auto-generated session configuration
├── conda_linux.yaml          # Conda environment (human_calib)
├── setup_models.sh           # VideoPose3D model download
├── demo/                     # Demo dataset (4 cameras, 100 frames)
├── input/                    # Place your calibration sessions here
├── output/                   # Calibration results
└── third_party/              # VideoPose3D submodule
```

---

## 6. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `libcuda.so not found` | WSL2 missing CUDA path | `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` (auto-set in calibrate.sh) |
| High MRE on one camera | Bad intrinsics (distortion) | Check distortion coefficients: k1/k2 should be in [-2, 2]. Values > 5 are likely wrong. |
| BA makes MRE worse | Regularization too strong | Auto-balanced lambda should handle this. If not, check if `objfun_multiview3d` is disabled. |
| `No valid orientations` | Too few visible frames | Lower `--conf_threshold` or use a different frame range where person is more visible. |
| MeTRAbs import error | Wrong conda env | MeTRAbs runs in `metrabs` env; `calibrate.sh` handles this via `conda run -n metrabs`. |
| OOM during BA | Too many frames | Pipeline auto-retries with larger `frame_skip`. |
| Poses not re-extracted | Cache hit | Delete `output/*/noise_1_0/2d_joint` and `3d_joint` to force re-extraction. |
| Same intrinsics work better than individual ones | Poor per-camera calibration | If cameras are the same model, try shared intrinsics as baseline. |

---

## Acknowledgments & Citations

This project builds upon [Extrinsic Camera Calibration From a Moving Person](https://github.com/kyotovision-public/extrinsic-camera-calibration-from-a-moving-person) (IROS 2022 / RA-L):

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

**MeTRAbs** — Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation:
- Original: [github.com/isarandi/metrabs](https://github.com/isarandi/metrabs)
- Fork used in this project: [github.com/flodelaplace/metrabs](https://github.com/flodelaplace/metrabs/tree/dev)

**RTMPose** — Real-Time Multi-Person Pose Estimation:
- [RTMLib](https://github.com/Tau-J/rtmlib) — Part of the [MMPose](https://github.com/open-mmlab/mmpose) ecosystem

**VideoPose3D** — 3D Human Pose Estimation in Video:
- [github.com/facebookresearch/VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
