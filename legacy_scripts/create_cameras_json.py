"""
create_cameras_json.py
----------------------
Crée le fichier cameras_G{gid:03d}.json requis par calib_linear.py et ba.py.
Ce fichier contient les paramètres INTRINSÈQUES de chaque caméra (matrice K).
Les paramètres extrinsèques (R, t) sont mis à l'identité car c'est justement
ce que le repo va estimer.

Usage:
    python create_cameras_json.py \
        --output_dir ./data/A001_P001_G001/raw_rtm \
        --gid 1 \
        --camera_ids 1 2 3 4 \
        --width 1088 \
        --height 1920 \
        --focal_length 1000   # en pixels (optionnel, estimé si absent)
        --cx 544              # centre optique x (optionnel, défaut = width/2)
        --cy 960              # centre optique y (optionnel, défaut = height/2)

Si tu ne connais pas la focale, une bonne estimation est:
    focal_length ≈ max(width, height)   (caméra standard ~70° FOV)
    focal_length ≈ max(width, height) * 1.2  (caméra légèrement zoomée)

Pour des smartphones typiquement : focal ≈ 1.2 * max(w, h)
"""

import argparse
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Crée cameras_G{gid}.json avec les intrinsèques pour le repo de calibration."
    )
    parser.add_argument("--output_dir", required=True,
                        help="Dossier de sortie (ex: ./data/A001_P001_G001/raw_rtm)")
    parser.add_argument("--gid", type=int, default=1, help="Group/Scene ID")
    parser.add_argument("--camera_ids", type=int, nargs="+", required=True,
                        help="Liste des IDs de caméra (ex: 1 2 3 4)")
    parser.add_argument("--width", type=int, required=True, help="Largeur des vidéos en pixels")
    parser.add_argument("--height", type=int, required=True, help="Hauteur des vidéos en pixels")
    parser.add_argument("--focal_length", type=float, default=None,
                        help="Focale en pixels (défaut: max(width, height) * 1.2)")
    parser.add_argument("--cx", type=float, default=None,
                        help="Centre optique x en pixels (défaut: width/2)")
    parser.add_argument("--cy", type=float, default=None,
                        help="Centre optique y en pixels (défaut: height/2)")
    args = parser.parse_args()

    # Valeurs par défaut
    fx = args.focal_length if args.focal_length else max(args.width, args.height) * 1.2
    fy = fx
    cx = args.cx if args.cx else args.width / 2.0
    cy = args.cy if args.cy else args.height / 2.0

    # Matrice intrinsèque K (3x3)
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)

    # Extrinsèques à l'identité (seront estimés par le repo)
    R_identity = np.eye(3, dtype=np.float64)
    t_zero = np.zeros(3, dtype=np.float64)

    n_cams = len(args.camera_ids)

    out = {
        "CAMID": args.camera_ids,
        "K": [K.tolist()] * n_cams,
        "R_w2c": [R_identity.tolist()] * n_cams,
        "t_w2c": [t_zero.tolist()] * n_cams,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"cameras_G{args.gid:03d}.json")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)

    print(f"Saved: {out_path}")
    print(f"  Cameras : {args.camera_ids}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Focal length: {fx:.1f} px")
    print(f"  Principal point: ({cx:.1f}, {cy:.1f})")

    # ---- Create skeleton_w placeholder ------------------------------------
    # skeleton_w_G{gid}.json stores 3D poses in world coordinates.
    # We don't have ground truth, so we create a placeholder with zeros.
    # The frame_indices must match the 2d/3d joint files (1-based, N frames).
    # We read the number of frames from the first 2d_joint JSON we can find.
    import glob
    joint_files = glob.glob(os.path.join(args.output_dir, "2d_joint", "*.json"))
    n_frames = 100  # fallback default
    if joint_files:
        with open(sorted(joint_files)[0]) as jf:
            jdata = json.load(jf)
            n_frames = len(jdata["data"])

    skeleton_out = {
        "skeleton": np.zeros((n_frames, 25, 3), dtype=np.float64).tolist(),
        "frame_indices": list(range(1, n_frames + 1)),
    }
    skel_path = os.path.join(args.output_dir, f"skeleton_w_G{args.gid:03d}.json")
    with open(skel_path, "w") as f:
        json.dump(skeleton_out, f, indent=2, ensure_ascii=True)
    print(f"Saved: {skel_path} (placeholder, {n_frames} frames)")
    print()
    print("NOTE: Si tu connais la vraie focale de tes caméras, relance avec --focal_length <valeur>")
    print("      La précision de la calibration extrinsèque en dépend !")


if __name__ == "__main__":
    main()

