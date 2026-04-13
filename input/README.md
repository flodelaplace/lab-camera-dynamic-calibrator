# Input Directory

Place your calibration sessions here. Each session should be a folder containing:

1. **Synchronized MP4 videos** — one per camera (file names are used as camera IDs)
2. **`Calib_scene.toml`** — intrinsic parameters for each camera

## Example structure

```
input/
  my_session/
    camera01.mp4
    camera02.mp4
    camera03.mp4
    Calib_scene.toml
```

## TOML format

```toml
[camera01]
name = "camera01"
size = [1920.0, 1080.0]
matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
distortions = [k1, k2, p1, p2]
fisheye = false
```

Videos and data files in this directory are git-ignored.
