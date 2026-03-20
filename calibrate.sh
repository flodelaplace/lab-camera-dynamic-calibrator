#!/bin/bash
# =============================================================================
# calibrate.sh — Pipeline complet de calibration extrinsèque
#
# Usage:
#   bash ./calibrate.sh <video_dir> <calib_toml> [output_dir] [device] [mode] [options]
#
# Arguments:
#   video_dir   : Dossier contenant les vidéos.
#   calib_toml  : Fichier TOML avec les paramètres intrinsèques.
#   output_dir  : (Optionnel) Dossier de sortie au format Axxx_Pxxx_Gxxx.
#   device      : (Optionnel) 'cpu' ou 'cuda'. Défaut: 'cuda'.
#   mode        : (Optionnel) 'lightweight', 'balanced', 'performance'. Défaut: 'balanced'.
#
# Options:
#   --height <h>      : Taille réelle de la personne en mètres (ex: 1.80) pour la mise à l'échelle.
#   --ref_frame <f>   : Frame de référence où la personne est droite pour l'orientation/échelle.
#   --start_frame <s> : Frame de début pour la calibration.
#   --end_frame <e>   : Frame de fin pour la calibration.
# =============================================================================

set -e  # Arrêter au moindre erreur

# --- Default values ---
DEVICE="cuda"
MODE="balanced"
PERSON_HEIGHT=""
REF_FRAME=""
START_FRAME=""
END_FRAME=""

# --- Parse arguments ---
# Positional arguments
VIDEO_DIR=$(realpath "${1}")
CALIB_TOML=$(realpath "${2}")
OUTPUT_DIR=${3:-"./data/session_$(date +%Y%m%d_%H%M%S)"}
# Shift past the main arguments to parse options
shift 3

# Parse optional named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --height) PERSON_HEIGHT="$2"; shift ;;
        --ref_frame) REF_FRAME="$2"; shift ;;
        --start_frame) START_FRAME="$2"; shift ;;
        --end_frame) END_FRAME="$2"; shift ;;
        cuda|cpu) DEVICE="$1" ;;
        lightweight|balanced|performance) MODE="$1" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")

# --- Extraire AID, PID, GID depuis le nom du dossier de sortie ---
BASENAME=$(basename "${OUTPUT_DIR}")
if [[ ${BASENAME} =~ A([0-9]+)_P([0-9]+)_G([0-9]+) ]]; then
    AID=$((10#${BASH_REMATCH[1]}))
    PID=$((10#${BASH_REMATCH[2]}))
    GID=$((10#${BASH_REMATCH[3]}))
else
    echo "WARNING: Le nom du dossier de sortie '${BASENAME}' ne suit pas le format Axxx_Pxxx_Gxxx. Using defaults."
    AID=1; PID=1; GID=1
fi

# Constantes internes
SUBSET="noise_1_0"
DATASET="MyDataset"
MODEL="pretrained_h36m_detectron_coco.bin"
FRAME_SKIP=10
LAMBDA1=1.
LAMBDA2=100000.

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Extrinsic Camera Calibration Pipeline               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Video Dir  : ${VIDEO_DIR}"
echo "║  Calib TOML : ${CALIB_TOML}"
echo "║  Output Dir : ${OUTPUT_DIR}"
echo "║  Session IDs: AID=${AID}, PID=${PID}, GID=${GID}"
echo "║  Device     : ${DEVICE}   Mode: ${MODE}"
if [ -n "$START_FRAME" ]; then
echo "║  Calib Range: Frames ${START_FRAME} to ${END_FRAME}"
fi
if [ -n "$PERSON_HEIGHT" ]; then
echo "║  Scaling    : Height=${PERSON_HEIGHT}m, Ref Frame=${REF_FRAME}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── Étape 1 : Extraction 2D poses ─────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/7] Extraction des poses 2D avec RTMPose..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  -> frame range: ${START_FRAME:-0} to ${END_FRAME:-<end>}"
python3 rtmlib_inference.py --video_dir "${VIDEO_DIR}" --output_dir "${OUTPUT_DIR}" --aid ${AID} --pid ${PID} --gid ${GID} --subset_name "${SUBSET}" --device ${DEVICE} --mode ${MODE} \
    $( [ -n "${START_FRAME}" ] && echo --start_frame "${START_FRAME}" ) \
    $( [ -n "${END_FRAME}" ] && echo --end_frame "${END_FRAME}" )


## ── Étape 1.5 : Créer le mappage des frames si une plage est spécifiée ────────
#if [ -n "$START_FRAME" ] && [ -n "$END_FRAME" ]; then
#    echo ""
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    echo "[1.5] Création du fichier de mappage des frames..."
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    python3 - << PYEOF
#import json, os, sys
#import numpy as np
#output_dir = "${OUTPUT_DIR}"
#subset = "${SUBSET}"
#gid = ${GID}
#start_frame = int("${START_FRAME}")
#end_frame = int("${END_FRAME}")
#
#num_frames = end_frame - start_frame + 1
#num_joints = 25 # Format OpenPose
#
#frame_indices = list(range(start_frame, end_frame + 1))
## Créer un squelette 3D factice pour la compatibilité avec les autres scripts
#dummy_skeleton = np.full((num_frames, num_joints, 3), None).tolist()
#
#mapping_data = {"frame_indices": frame_indices, "skeleton": dummy_skeleton}
#
#output_path = os.path.join(output_dir, subset, f"skeleton_w_G{gid:03d}.json")
#os.makedirs(os.path.dirname(output_path), exist_ok=True)
#with open(output_path, "w") as f:
#    json.dump(mapping_data, f)
#print(f"  -> Fichier de mappage compatible créé : {output_path}")
#PYEOF
#fi


# ── Étape 2 : Créer cameras_Gxxx.json ─────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/7] Lecture des intrinsèques depuis le TOML..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CAM_NAMES=()
while IFS= read -r -d $'\0' file; do BNAME=$(basename "$file"); CAM_NAMES+=("${BNAME%.*}"); done < <(find "${VIDEO_DIR}" -maxdepth 1 \( -name "*.mp4" -o -name "*.MP4" \) -print0 | sort -zV)
python3 create_cameras_from_toml.py --toml "${CALIB_TOML}" --output_dir "${OUTPUT_DIR}/${SUBSET}" --gid ${GID} --cam_names "${CAM_NAMES[@]}"


# ── Étape 2.5 (Anciennement 1.5) : Créer le mappage des frames ────────
if [ -n "$START_FRAME" ] && [ -n "$END_FRAME" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[2.5] Création du fichier de mappage des frames..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 - << PYEOF
import json, os, sys
import numpy as np
output_dir = "${OUTPUT_DIR}"
subset = "${SUBSET}"
gid = ${GID}
start_frame = int("${START_FRAME}")
end_frame = int("${END_FRAME}")

num_frames = end_frame - start_frame + 1
num_joints = 25 # Format OpenPose

# RTMPose conserve les numéros de frames originaux (ex: 1500 à 2499)
frame_indices = list(range(start_frame, end_frame + 1))
dummy_skeleton = np.full((num_frames, num_joints, 3), None).tolist()

mapping_data = {"frame_indices": frame_indices, "skeleton": dummy_skeleton}

output_path = os.path.join(output_dir, subset, f"skeleton_w_G{gid:03d}.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(mapping_data, f)
print(f"  -> Fichier de mappage compatible créé : {output_path}")
PYEOF
fi

# ── Étape 3 : Mise à jour de la configuration ────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/7] Mise à jour de la configuration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - << PYEOF
import yaml, json, sys, os, glob, re
script_dir = os.getcwd() # Ou utiliser un chemin plus robuste si nécessaire
output_dir, subset, calib_toml, aid, pid, gid = "${OUTPUT_DIR}", "${SUBSET}", "${CALIB_TOML}", ${AID}, ${PID}, ${GID}
cam_file = os.path.join(output_dir, subset, f"cameras_G{gid:03d}.json")
with open(cam_file) as f: n_cams = len(json.load(f)["CAMID"])
jfiles = sorted(glob.glob(os.path.join(output_dir, subset, "2d_joint", f"A{aid:03d}_P{pid:03d}_G{gid:03d}_C*.json")))
with open(jfiles[0]) as f: n_frames = len(json.load(f)["data"])
with open("./config/config.yaml") as f: config = yaml.safe_load(f)
config["MyDataset"] = {"width": 1920, "height": 1080, "scale": 1, "frame_rate": 30, "camera_ids": list(range(1, n_cams + 1)), "available_joints": list(range(15)), "ransac_th_2d": 200.0, "ransac_th_3d": 1.0}
config.update({"aid": aid, "pid": pid, "gid": gid})
with open("./config/config.yaml", "w") as f: yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print(f"Config updated: {n_cams} cameras, {n_frames} frames")
PYEOF

# ── Étape 4 : Lifting 2D -> 3D ────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/7] Lifting 2D -> 3D avec VideoPose3D..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash ./inference.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${MODEL} ${DATASET}

# ── Étape 5 : Calibration extrinsèque ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/7] Calibration extrinsèque..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  → Calibration linéaire par chunks..."

# Les données sont déjà rognées, on calibre simplement sur la totalité des frames disponibles en local (index 0 à N-1)
bash ./calib_linear.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${SUBSET} ${FRAME_SKIP} ${DATASET}

# Verify linear calibration result exists
FINAL_LINEAR_JSON="${OUTPUT_DIR}/results/linear_1_0.json"
if [ ! -f "${FINAL_LINEAR_JSON}" ]; then
    echo "ERROR: Linear calibration result not found: ${FINAL_LINEAR_JSON}"
    echo "Aborting bundle adjustment. Please check calib_linear.sh output and logs."
    exit 1
fi
echo "  → Bundle Adjustment (linéaire)..."
bash ./ba.sh "${OUTPUT_DIR}" ${AID} ${PID} ${GID} ${FRAME_SKIP} ${LAMBDA1} ${LAMBDA2} linear_1_0 ${DATASET} false true

# ── Étape 6 : Évaluation et Visualisation ─────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/7] Évaluation et Visualisation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
declare -A MRE_SCORES
BEST_MRE=99999
BEST_CALIB=""
for CALIB in linear_1_0 linear_1_0_ba; do
    if [ -f "${OUTPUT_DIR}/results/${CALIB}.json" ]; then
        OUTPUT=$(python3 evaluate_calibration.py --prefix "${OUTPUT_DIR}" --calib "${CALIB}" --video_dir "${VIDEO_DIR}" --visualize \
            $( [ -n "${START_FRAME}" ] && echo --start_frame "${START_FRAME}" ))
        echo "${OUTPUT}"
        MRE=$(echo "${OUTPUT}" | grep "Global MRE" | awk '{print $4}')
        if [ -n "$MRE" ]; then
            MRE_SCORES[$CALIB]=$MRE
            if (( $(echo "$MRE < $BEST_MRE" | bc -l) )); then BEST_MRE=$MRE; BEST_CALIB=$CALIB; fi
        fi
        echo "  → Visualisation 3D pour ${CALIB}..."
        python3 visualize_results.py --prefix "${OUTPUT_DIR}" --subset "${SUBSET}" --calib "${CALIB}" --dataset ${DATASET} --output "${OUTPUT_DIR}/results/camera/visu_3d_${CALIB}.gif" || echo "  ⚠ Visu ${CALIB} failed"
    fi
done

# ── Étape 7 : Mise à l'échelle et Orientation (Optionnel) ─────────────────────
FINAL_TOML_PATH="${OUTPUT_DIR}/results/Calib_scene_calibrated.toml"
FINAL_CALIB_NAME=""
if [ -n "$PERSON_HEIGHT" ] && [ -n "$REF_FRAME" ] && [ -n "$BEST_CALIB" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[7/7] Mise à l'échelle, Orientation et Visualisation Finale..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # --- Mappage de la frame de référence ---
    MAPPED_REF_FRAME=$REF_FRAME
    if [ -n "$START_FRAME" ]; then
        MAPPED_REF_FRAME=$((REF_FRAME - START_FRAME))
        echo "  -> Frame de référence re-mappée de ${REF_FRAME} à l'indice ${MAPPED_REF_FRAME} pour correspondre aux données rognées."
    fi

    python3 scale_scene.py \
        --prefix "${OUTPUT_DIR}" \
        --calib "${BEST_CALIB}" \
        --height ${PERSON_HEIGHT} \
        --frame_idx ${MAPPED_REF_FRAME} \
        --input_toml "${CALIB_TOML}" \
        --export_toml "${FINAL_TOML_PATH}" \
        --video_dir "${VIDEO_DIR}"

    FINAL_CALIB_NAME="${BEST_CALIB}_oriented_scaled"
    if [ -f "${OUTPUT_DIR}/results/${FINAL_CALIB_NAME}.json" ]; then
        echo "  → Visualisation finale..."
        python3 visualize_results.py \
            --prefix  "${OUTPUT_DIR}" \
            --subset  "${SUBSET}" \
            --calib   "${FINAL_CALIB_NAME}" \
            --dataset ${DATASET} \
            --output  "${OUTPUT_DIR}/results/camera/visu_3d_FINAL.gif"
    fi
fi

# ── Résumé final ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      ✅  DONE !                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Résultats dans : ${OUTPUT_DIR}/results/"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  📊 Tableau Récapitulatif des MRE (Mean Reprojection Error)  ║"
echo "║──────────────────────────────────────────────────────────────║"
printf "║ %-25s | %-s\n" "Méthode" "MRE (pixels)"
echo "║──────────────────────────────────────────────────────────────║"
for CALIB in "${!MRE_SCORES[@]}"; do
    if [ "$CALIB" == "$BEST_CALIB" ]; then
        printf "║⭐ %-23s | %-s\n" "$CALIB" "${MRE_SCORES[$CALIB]}"
    else
        printf "║  %-24s | %-s\n" "$CALIB" "${MRE_SCORES[$CALIB]}"
    fi
done
if [ -n "$FINAL_CALIB_NAME" ]; then
    echo "║──────────────────────────────────────────────────────────────║"
    echo "║  Fichier TOML final généré :                                 ║"
    printf "║    results/%s\n" "$(basename ${FINAL_TOML_PATH})"
    echo "║  Visualisation finale :                                      ║"
    echo "║    results/camera/visu_3d_FINAL.gif                          ║"
else
    echo "║──────────────────────────────────────────────────────────────║"
    echo "║ Pour générer un TOML final, relancez avec les options :      ║"
    echo "║   --height <h> --ref_frame <f>                               ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
