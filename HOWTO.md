# Usage Guide

This guide explains how to calibrate camera extrinsics from your own videos using this pipeline.

### 1. Prepare your data

1. Place your synchronized videos (`.mp4`, `.avi`, `.mov`) in a directory. Cameras must be static; the subject must move across the capture volume.
2. Provide a `Calib_scene.toml` file with intrinsic parameters per camera, matching the [Pose2Sim](https://github.com/perfanalytics/pose2sim) format.

> ⚠ Intrinsic quality is the #1 driver of final MRE. Distortion coefficients k1, k2 above ~5 in absolute value are almost certainly wrong — re-calibrate that camera before continuing.

*(A ready-to-run `demo/` folder is provided: 4 synchronized videos + `Calib_scene.toml`.)*

Optional but useful: a `<video>.dropped.json` sidecar next to each video, listing absolute frame indices that should be ignored everywhere (corrupted / black frames). The auto outlier-frame drop step also writes these automatically — see the main README for details.

### 2. Run the full pipeline

`scripts/calibrate.sh` orchestrates the 7-step pipeline (pose extraction → intrinsics loading → linear init → auto outlier-frame drop → BA → MRE evaluation → scaling).

**Recommended: MeTRAbs path** (direct metric 3D, Procrustes init, much higher accuracy):

```bash
bash scripts/calibrate.sh \
    demo \
    demo/Calib_scene.toml \
    output/demo_metrabs \
    cuda balanced \
    --pose_engine metrabs \
    --height 1.78 \
    --ref_frame 5
```

**RTMPose + VideoPose3D path** (legacy two-step, kept for comparison):

```bash
bash scripts/calibrate.sh \
    demo \
    demo/Calib_scene.toml \
    output/demo_rtmpose \
    cuda balanced \
    --height 1.78 \
    --ref_frame 5
```

#### Positional arguments

| Position | Description |
|----------|-------------|
| 1 | Folder containing the synchronized videos |
| 2 | Path to the intrinsics TOML |
| 3 | Output directory (will be created) |
| 4 | `cuda` or `cpu` (optional, default `cuda`) |
| 5 | `lightweight` / `balanced` / `performance` (optional, RTMPose only) |

#### Named flags

| Flag | Default | Effect |
|------|---------|--------|
| `--pose_engine <eng>` | `rtmpose` | `metrabs` (recommended) or `rtmpose` |
| `--height <m>` | — | Subject height in **meters** (e.g. `1.84`). Enables step 7 (scaling + orientation). |
| `--ref_frame <n>` | — | Frame where the subject is standing straight, feet flat. Used to define the floor and to scale to metric units. Must be inside `[start_frame, end_frame]`. |
| `--start_frame <n>` | `0` | First frame to process. |
| `--end_frame <n>` | last | Last frame to process. |
| `--frame_skip <n>` | `10` | Subsample interval for BA. Lower = denser optimization, slower. `5` is a good default with MeTRAbs. |
| `--conf_threshold <t>` | `0.5` | Minimum 2D keypoint confidence. Lower = more data, more noise. |
| `--ref_cam <id>` | *(auto)* | 1-indexed CAM ID to force as Procrustes reference. Default: auto-select the camera with the lowest mean Procrustes residual. |
| `--no_auto_outlier_drop` | off | Disable the per-camera outlier-frame drop step between linear and BA. |
| `--outlier_abs_px <p>` | `50` | Absolute reproj threshold for the outlier drop. |
| `--outlier_x_median <m>` | `5` | Multiplier above per-camera median for the outlier drop (frame must exceed **both** thresholds to be dropped). |
| `--save_video` | off | Save 2D pose overlay video (RTMPose only). |

#### Full real-world example

```bash
bash scripts/calibrate.sh \
    input/my_session \
    input/my_session/Calib_scene.toml \
    output/my_session \
    cuda balanced \
    --pose_engine metrabs \
    --start_frame 650 --end_frame 1500 \
    --ref_frame 1415 \
    --height 1.84 \
    --frame_skip 5
```

This processes frames 650–1500, calibrates with MeTRAbs, scales the scene using a 1.84 m subject standing at frame 1415.

### 3. Check the results

The pipeline prints a summary table at the end with the MRE (Mean Reprojection Error) for each calibration stage; the best one is starred. All outputs land in `<output_dir>/results/`:

| File | Description |
|------|-------------|
| `Calib_scene_calibrated.toml` | Final calibration (metric, gravity-aligned) — ready for Pose2Sim / OpenCap / OpenSim. |
| `3d_skeleton_FINAL.trc` | Triangulated 3D skeleton in TRC format. |
| `camera/visu_3d_FINAL.gif` | Animated 3D viz with live MRE metrics overlaid (see the README hero image). |
| `camera/visu_3d_linear_1_0.gif` | Intermediate viz after the linear init only. |
| `camera/visu_3d_linear_1_0_ba.gif` | Intermediate viz after BA, before scaling. |
| `MRE_visualizations/` | Per-camera best/worst reprojection images — diagnostic for any cam that stays high after BA. |
| `ba_cost_live_iter*.png` | Bundle Adjustment convergence curves. |

### 4. Diagnose / improve

If one camera lags behind the others (e.g. MRE 9 px while the rest are at 5 px):

- Look at its `MRE_visualizations/<calib>/camX_worst.png` — if the reprojected points are systematically offset, the camera's K is wrong; re-calibrate it.
- Check the Procrustes log line for that camera in the linear init. **Low Procrustes residual (≤ ~100 mm) but high MRE ⇒ intrinsics issue** (the 3D shape from MeTRAbs is fine, the projection back to 2D is biased by a wrong K).
- The auto-outlier drop writes per-video `<video>.dropped.json` files in your input folder — check them to see which frames were flagged.

If the auto-selected reference camera doesn't seem right (e.g. you know cam 3 has the cleanest view of the subject), force it:

```bash
... --ref_cam 3
```

### 5. Caching

When using MeTRAbs, pose extraction results are cached in the output dir. Re-runs with the **same frame range** skip inference automatically (~2 min/cam saved).

> The cache only checks the frame range, **not** the intrinsics. If you change the TOML, delete the cached poses to force re-extraction:
> ```bash
> rm -rf output/my_session/noise_1_0/2d_joint output/my_session/noise_1_0/3d_joint
> ```

For deeper details (BA design, joint formats, etc.) see the main [README](README.md).
