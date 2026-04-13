"""Convertit un fichier de calibration TOML en faisant une rotation de 90deg

Usage:
  python utils/convert_calib_rotation.py --in in.toml --out out.toml --dir cw

Options for --dir: cw (clockwise 90 deg), ccw (counter-clockwise 90 deg)

Fonctionnalités:
 - inverse la taille (size)
 - met à jour la matrice intrinsèque `matrix`
 - met à jour `distortions` (OpenCV: [k1,k2,p1,p2])
 - préserve les autres champs
 - supporte plusieurs caméras dans le TOML

Ce script suppose que la clé des caméras est un tableau de tables ou un dictionnaire
contenant pour chaque caméra les champs: name, size, matrix, distortions, etc.
"""
import argparse
import toml
import copy
import sys
from pathlib import Path


def rotate_intrinsics(matrix, size, direction):
    """Rotate intrinsic matrix by 90 degrees.

    matrix: 3x3 list
    size: [width, height]
    direction: 'cw' or 'ccw'
    Returns new_matrix, new_size
    """
    w, h = float(size[0]), float(size[1])
    # matrix elements
    f_x = float(matrix[0][0])
    f_y = float(matrix[1][1])
    c_x = float(matrix[0][2])
    c_y = float(matrix[1][2])

    # After rotation, focal lengths swap
    fx_p = f_y
    fy_p = f_x

    if direction == 'cw':
        # clockwise 90 deg: new_cx = old_height - old_cy ; new_cy = old_cx
        cx_p = h - c_y
        cy_p = c_x
    else:
        # ccw: new_cx = old_cy ; new_cy = old_width - old_cx
        cx_p = c_y
        cy_p = w - c_x

    new_matrix = [
        [float(fx_p), 0.0, float(cx_p)],
        [0.0, float(fy_p), float(cy_p)],
        [0.0, 0.0, 1.0],
    ]
    new_size = [h, w]
    return new_matrix, new_size


def rotate_distortions(dist, direction):
    """Rotate distortions [k1,k2,p1,p2]
    k1,k2 unchanged. p1,p2 change depending on rotation.
    """
    if dist is None:
        return dist
    if len(dist) < 4:
        # nothing to do
        return dist
    k1, k2, p1, p2 = float(dist[0]), float(dist[1]), float(dist[2]), float(dist[3])
    if direction == 'cw':
        p1_p = p2
        p2_p = -p1
    else:
        p1_p = -p2
        p2_p = p1
    return [k1, k2, p1_p, p2_p] + [float(x) for x in dist[4:]]


def process_camera(cam, direction):
    cam = copy.deepcopy(cam)
    # size: expect [width, height]
    if 'size' in cam and 'matrix' in cam:
        try:
            new_mat, new_size = rotate_intrinsics(cam['matrix'], cam['size'], direction)
            cam['matrix'] = new_mat
            cam['size'] = [float(new_size[0]), float(new_size[1])]
        except Exception as e:
            print(f"Warning: failed to rotate intrinsics for camera {cam.get('name','?')}: {e}")

    if 'distortions' in cam:
        try:
            cam['distortions'] = rotate_distortions(cam['distortions'], direction)
        except Exception as e:
            print(f"Warning: failed to rotate distortions for camera {cam.get('name','?')}: {e}")

    return cam


def main():
    parser = argparse.ArgumentParser(description='Convert calibration TOML by rotating cameras 90deg')
    parser.add_argument('--in', dest='infile', required=True, help='Input toml file')
    parser.add_argument('--out', dest='outfile', required=True, help='Output toml file')
    parser.add_argument('--dir', dest='direction', choices=['cw', 'ccw'], default='cw', help='Rotation direction: cw or ccw')
    parser.add_argument('--dry-run', action='store_true', help='Print changes without writing file')
    args = parser.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    if not in_path.exists():
        print(f"Input file {in_path} not found", file=sys.stderr)
        sys.exit(2)

    data = toml.load(str(in_path))

    # The structure of TOML may vary. We'll try to find camera tables.
    # Common formats:
    # 1) top-level table per camera (keys are camera ids) -> dict of dicts
    # 2) cameras = [ {..}, {..} ] -> list

    modified = False

    # Case 1: top-level keys that look like cameras: keys whose value is a table with 'matrix' or 'size'
    for k, v in list(data.items()):
        if isinstance(v, dict) and ('matrix' in v or 'size' in v or 'distortions' in v):
            data[k] = process_camera(v, args.direction)
            modified = True

    # Case 2: cameras list
    if 'cameras' in data and isinstance(data['cameras'], list):
        newcams = []
        for cam in data['cameras']:
            newcams.append(process_camera(cam, args.direction))
            modified = True
        data['cameras'] = newcams

    if not modified:
        print("No camera entries found to modify. Make sure TOML contains 'matrix' or 'size' fields.")

    if args.dry_run:
        print(toml.dumps(data))
    else:
        toml_string = toml.dumps(data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(toml_string)
        print(f"Wrote rotated calibration to {out_path}")


if __name__ == '__main__':
    main()

