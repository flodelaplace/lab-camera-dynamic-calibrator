"""
create_cameras_from_toml.py
----------------------------
Crée cameras_G{gid}.json et skeleton_w_G{gid}.json à partir d'un fichier
de calibration au format TOML (ex: Pose2Sim / AniPose).

Ce script recherche des sections dans le TOML qui correspondent aux noms
des vidéos fournies.

Usage:
    python create_cameras_from_toml.py \
        --toml       ./Calibration.toml \
        --output_dir ./data/A001_P001_G001/raw_rtm \
        --gid 1 \
        --cam_names "video1" "video2" "video3"

Le fichier TOML doit avoir des sections [video_name] avec:
    matrix       = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    rotation     = [rx, ry, rz]   <- vecteur de Rodrigues
    translation  = [tx, ty, tz]
    size         = [width, height]
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import cv2

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def parse_toml(path):
    """Parse a TOML file, falling back to manual parsing if tomllib unavailable."""
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)

    import re
    data = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^\[(.+)\]$', line)
            if m:
                current = m.group(1)
                data[current] = {}
                continue
            if current and '=' in line:
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()
                try:
                    data[current][key] = eval(val)
                except Exception:
                    data[current][key] = val.strip('"')
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un fichier TOML de calibration en cameras_G{gid}.json"
    )
    parser.add_argument("--toml", required=True, help="Chemin vers le fichier .toml")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie")
    parser.add_argument("--gid", type=int, default=1, help="Group/Scene ID")
    parser.add_argument("--cam_names", nargs='+', required=True, help="Liste des noms de base des vidéos (sans extension)")
    args = parser.parse_args()

    # ---- Lire le TOML -------------------------------------------------------
    data = parse_toml(args.toml)

    print(f"Recherche des sections pour les caméras: {args.cam_names}")

    # ---- Extraire K, R, t pour chaque caméra --------------------------------
    cam_ids, K_list, R_list, t_list = [], [], [], []

    for i, cam_name in enumerate(args.cam_names, start=1):
        if cam_name not in data:
            print(f"ERROR: La section '[{cam_name}]' n'a pas été trouvée dans le fichier TOML {args.toml}", file=sys.stderr)
            sys.exit(1)
        
        sec = data[cam_name]
        cam_ids.append(i)

        K = np.array(sec["matrix"], dtype=np.float64)
        K_list.append(K)

        rvec = np.array(sec["rotation"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        R_list.append(R)

        t = np.array(sec["translation"], dtype=np.float64)
        t_list.append(t)

        print(f"  -> Trouvé '{cam_name}' (Cam ID {i})")

    # ---- Sauvegarder cameras_G{gid}.json ------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    cam_path = os.path.join(args.output_dir, f"cameras_G{args.gid:03d}.json")

    out = {
        "CAMID": cam_ids,
        "K": [k.tolist() for k in K_list],
        "R_w2c": [R.tolist() for R in R_list],
        "t_w2c": [t.tolist() for t in t_list],
    }
    with open(cam_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"\nSaved: {cam_path}")

    # ---- Sauvegarder skeleton_w_G{gid}.json (placeholder) -------------------
    joint_files = glob.glob(os.path.join(args.output_dir, "2d_joint", "*.json"))
    n_frames = 100
    if joint_files:
        with open(sorted(joint_files)[0]) as jf:
            jdata = json.load(jf)
            n_frames = len(jdata["data"])

    skel_out = {
        "skeleton": np.zeros((n_frames, 25, 3), dtype=np.float64).tolist(),
        "frame_indices": list(range(1, n_frames + 1)),
    }
    skel_path = os.path.join(args.output_dir, f"skeleton_w_G{args.gid:03d}.json")
    with open(skel_path, "w") as f:
        json.dump(skel_out, f, indent=2, ensure_ascii=True)
    print(f"Saved: {skel_path} (placeholder, {n_frames} frames)")

if __name__ == "__main__":
    main()
