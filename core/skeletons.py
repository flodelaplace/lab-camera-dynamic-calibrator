"""Skeleton format definitions: joint indices, bone connectivity, format auto-detection.

Three skeleton formats are supported:
- OpenPose-25 (OP_KEY / OP_BONE): legacy format used by RTMPose + VideoPose3D
- MeTRAbs calib-26 (METRABS_KEY / METRABS_BONE): subset extracted from bml_movi_87
- bml_movi_87 (BML87_KEY / BML87_BONE): full MeTRAbs output skeleton
"""
import numpy as np


# ---------------------------------------------------------------------------
# OpenPose-25 (legacy, RTMPose + VideoPose3D path)
# ---------------------------------------------------------------------------
OP_KEY = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24,
}

COCO_KEY = {
    "Nose": 0,
    "L_Eye": 1,
    "R_Eye": 2,
    "L_Ear": 3,
    "R_Ear": 4,
    "L_Shoulder": 5,
    "R_Shoulder": 6,
    "L_Elbow": 7,
    "R_Elbow": 8,
    "L_Wrist": 9,
    "R_Wrist": 10,
    "L_Hip": 11,
    "R_Hip": 12,
    "L_Knee": 13,
    "R_Knee": 14,
    "L_Ankle": 15,
    "R_Ankle": 16,
}

H36M32_KEY = {
    "Pelvis": 0,
    "R_Hip": 1,
    "R_Knee": 2,
    "R_Ankle": 3,
    "L_Hip": 6,
    "L_Knee": 7,
    "L_Ankle": 8,
    "Spin": 12,
    "Thorax": 13,
    "Nose": 14,
    "Head": 15,
    "L_Shoulder": 17,
    "L_Elbow": 18,
    "L_Wrist": 19,
    "R_Shoulder": 25,
    "R_Elbow": 26,
    "R_Wrist": 27,
}

op_to_coco = {
    "Nose": "Nose",
    "LEye": "L_Eye",
    "REye": "R_Eye",
    "LEar": "L_Ear",
    "REar": "R_Ear",
    "LShoulder": "L_Shoulder",
    "RShoulder": "R_Shoulder",
    "LElbow": "L_Elbow",
    "RElbow": "R_Elbow",
    "LWrist": "L_Wrist",
    "RWrist": "R_Wrist",
    "LHip": "L_Hip",
    "RHip": "R_Hip",
    "LKnee": "L_Knee",
    "RKnee": "R_Knee",
    "LAnkle": "L_Ankle",
    "RAnkle": "R_Ankle",
}

H36M17_KEY = {
    "Pelvis": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Spin": 7,
    "Thorax": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}


OP_BONE = np.array(
    [
        [OP_KEY["MidHip"], OP_KEY["Neck"]],
        [OP_KEY["RShoulder"], OP_KEY["LShoulder"]],
        [OP_KEY["RShoulder"], OP_KEY["RElbow"]],
        [OP_KEY["LShoulder"], OP_KEY["LElbow"]],
        [OP_KEY["RWrist"], OP_KEY["RElbow"]],
        [OP_KEY["LWrist"], OP_KEY["LElbow"]],
        [OP_KEY["RHip"], OP_KEY["MidHip"]],
        [OP_KEY["LHip"], OP_KEY["MidHip"]],
        [OP_KEY["RHip"], OP_KEY["RKnee"]],
        [OP_KEY["RKnee"], OP_KEY["RAnkle"]],
        [OP_KEY["LHip"], OP_KEY["LKnee"]],
        [OP_KEY["LKnee"], OP_KEY["LAnkle"]],
    ],
    dtype=int,
)


OP_KEY_SUB = np.sort(np.unique(OP_BONE.flatten()))


def mk_bone_sub(bones, subjoints):
    k = subjoints
    v = np.arange(len(subjoints))
    sidx = k.argsort()
    return v[sidx[np.searchsorted(k, bones, sorter=sidx)]]


OP_BONE_SUB = mk_bone_sub(OP_BONE, OP_KEY_SUB)


# ---------------------------------------------------------------------------
# MeTRAbs 26-joint skeleton for calibration (bml_movi_87 subset)
# 20 virtual joint centers + backneck + sternum + 4 foot markers
# ---------------------------------------------------------------------------
METRABS_KEY = {
    "head":     0,
    "backneck": 1,
    "thor":     2,
    "sternum":  3,
    "pelv":     4,
    "mhip":     5,
    "lsho":     6,
    "rsho":     7,
    "lelb":     8,
    "relb":     9,
    "lwri":     10,
    "rwri":     11,
    "lhan":     12,
    "rhan":     13,
    "lhip":     14,
    "rhip":     15,
    "lkne":     16,
    "rkne":     17,
    "lank":     18,
    "rank":     19,
    "lfoo":     20,
    "rfoo":     21,
    "lhee":     22,
    "rhee":     23,
    "ltoe":     24,
    "rtoe":     25,
}

# Indices in bml_movi_87 corresponding to each METRABS_KEY joint (ordered by local index)
METRABS_BML87_INDICES = [67, 0, 70, 3, 69, 68, 76, 84, 72, 80, 77, 85, 74, 82, 73, 81, 75, 83, 71, 79, 78, 86, 21, 52, 23, 54]

MK = METRABS_KEY  # shorthand

METRABS_BONE = np.array(
    [
        # Spine (4 bones — vs 1 in OP_BONE)
        [MK["mhip"],    MK["pelv"]],
        [MK["pelv"],    MK["thor"]],
        [MK["thor"],    MK["backneck"]],
        [MK["backneck"],MK["head"]],
        # Torso cross-constraints
        [MK["sternum"], MK["thor"]],
        [MK["lsho"],    MK["rsho"]],      # shoulder width
        [MK["lhip"],    MK["rhip"]],       # hip width
        # Shoulder connections
        [MK["thor"],    MK["lsho"]],
        [MK["thor"],    MK["rsho"]],
        # Hip connections
        [MK["pelv"],    MK["lhip"]],
        [MK["pelv"],    MK["rhip"]],
        # Left arm (3 bones)
        [MK["lsho"],    MK["lelb"]],
        [MK["lelb"],    MK["lwri"]],
        [MK["lwri"],    MK["lhan"]],
        # Right arm (3 bones)
        [MK["rsho"],    MK["relb"]],
        [MK["relb"],    MK["rwri"]],
        [MK["rwri"],    MK["rhan"]],
        # Left leg (3 bones)
        [MK["lhip"],    MK["lkne"]],
        [MK["lkne"],    MK["lank"]],
        [MK["lank"],    MK["lfoo"]],
        # Right leg (3 bones)
        [MK["rhip"],    MK["rkne"]],
        [MK["rkne"],    MK["rank"]],
        [MK["rank"],    MK["rfoo"]],
        # Feet (4 bones — new)
        [MK["lank"],    MK["lhee"]],
        [MK["lhee"],    MK["ltoe"]],
        [MK["rank"],    MK["rhee"]],
        [MK["rhee"],    MK["rtoe"]],
    ],
    dtype=int,
)  # 27 bones total (vs 12 for OP_BONE)

METRABS_KEY_SUB = np.sort(np.unique(METRABS_BONE.flatten()))
METRABS_BONE_SUB = mk_bone_sub(METRABS_BONE, METRABS_KEY_SUB)


# ---------------------------------------------------------------------------
# Full bml_movi_87 skeleton (all 87 joints from MeTRAbs)
# ---------------------------------------------------------------------------
BML87_KEY = {
    "backneck": 0, "upperback": 1, "clavicle": 2, "sternum": 3, "umbilicus": 4,
    "lfronthead": 5, "lbackhead": 6, "lback": 7, "lshom": 8, "lupperarm": 9,
    "lelbm": 10, "lforearm": 11, "lwrithumbside": 12, "lwripinkieside": 13,
    "lfin": 14, "lasis": 15, "lpsis": 16, "lfrontthigh": 17, "lthigh": 18,
    "lknem": 19, "lankm": 20, "lhee": 21, "lfifthmetatarsal": 22, "ltoe": 23,
    "lcheek": 24, "lbreast": 25, "lelbinner": 26, "lwaist": 27, "lthumb": 28,
    "lfrontinnerthigh": 29, "linnerknee": 30, "lshin": 31, "lfirstmetatarsal": 32,
    "lfourthtoe": 33, "lscapula": 34, "lbum": 35,
    "rfronthead": 36, "rbackhead": 37, "rback": 38, "rshom": 39, "rupperarm": 40,
    "relbm": 41, "rforearm": 42, "rwrithumbside": 43, "rwripinkieside": 44,
    "rfin": 45, "rasis": 46, "rpsis": 47, "rfrontthigh": 48, "rthigh": 49,
    "rknem": 50, "rankm": 51, "rhee": 52, "rfifthmetatarsal": 53, "rtoe": 54,
    "rcheek": 55, "rbreast": 56, "relbinner": 57, "rwaist": 58, "rthumb": 59,
    "rfrontinnerthigh": 60, "rinnerknee": 61, "rshin": 62, "rfirstmetatarsal": 63,
    "rfourthtoe": 64, "rscapula": 65, "rbum": 66,
    "head": 67, "mhip": 68, "pelv": 69, "thor": 70,
    "lank": 71, "lelb": 72, "lhip": 73, "lhan": 74, "lkne": 75,
    "lsho": 76, "lwri": 77, "lfoo": 78,
    "rank": 79, "relb": 80, "rhip": 81, "rhan": 82, "rkne": 83,
    "rsho": 84, "rwri": 85, "rfoo": 86,
}

B = BML87_KEY  # shorthand

BML87_BONE = np.array([
    # === Official MeTRAbs edges (19) ===
    [B["head"],  B["thor"]],
    [B["mhip"],  B["pelv"]],
    [B["mhip"],  B["lhip"]],
    [B["mhip"],  B["rhip"]],
    [B["pelv"],  B["thor"]],
    [B["thor"],  B["lsho"]],
    [B["thor"],  B["rsho"]],
    [B["lank"],  B["lkne"]],
    [B["lank"],  B["lfoo"]],
    [B["lelb"],  B["lsho"]],
    [B["lelb"],  B["lwri"]],
    [B["lhip"],  B["lkne"]],
    [B["lhan"],  B["lwri"]],
    [B["rank"],  B["rkne"]],
    [B["rank"],  B["rfoo"]],
    [B["relb"],  B["rsho"]],
    [B["relb"],  B["rwri"]],
    [B["rhip"],  B["rkne"]],
    [B["rhan"],  B["rwri"]],
    # === Spine & torso detail ===
    [B["thor"],  B["backneck"]],
    [B["backneck"], B["head"]],
    [B["thor"],  B["sternum"]],
    [B["thor"],  B["clavicle"]],
    [B["pelv"],  B["umbilicus"]],
    [B["thor"],  B["upperback"]],
    # === Head markers ===
    [B["head"],  B["lfronthead"]],
    [B["head"],  B["rfronthead"]],
    [B["head"],  B["lbackhead"]],
    [B["head"],  B["rbackhead"]],
    [B["head"],  B["lcheek"]],
    [B["head"],  B["rcheek"]],
    # === Shoulder/scapula markers ===
    [B["lsho"],  B["lshom"]],
    [B["lsho"],  B["lscapula"]],
    [B["rsho"],  B["rshom"]],
    [B["rsho"],  B["rscapula"]],
    [B["lsho"],  B["rsho"]],  # shoulder width
    # === Arm detail ===
    [B["lelb"],  B["lelbm"]],
    [B["lelb"],  B["lelbinner"]],
    [B["lwri"],  B["lwrithumbside"]],
    [B["lwri"],  B["lwripinkieside"]],
    [B["lhan"],  B["lfin"]],
    [B["lhan"],  B["lthumb"]],
    [B["relb"],  B["relbm"]],
    [B["relb"],  B["relbinner"]],
    [B["rwri"],  B["rwrithumbside"]],
    [B["rwri"],  B["rwripinkieside"]],
    [B["rhan"],  B["rfin"]],
    [B["rhan"],  B["rthumb"]],
    # === Upper arm ===
    [B["lsho"],  B["lupperarm"]],
    [B["lelb"],  B["lforearm"]],
    [B["rsho"],  B["rupperarm"]],
    [B["relb"],  B["rforearm"]],
    # === Torso markers ===
    [B["thor"],  B["lbreast"]],
    [B["thor"],  B["rbreast"]],
    [B["pelv"],  B["lback"]],
    [B["pelv"],  B["rback"]],
    [B["pelv"],  B["lwaist"]],
    [B["pelv"],  B["rwaist"]],
    [B["lhip"],  B["lasis"]],
    [B["lhip"],  B["lpsis"]],
    [B["lhip"],  B["lbum"]],
    [B["rhip"],  B["rasis"]],
    [B["rhip"],  B["rpsis"]],
    [B["rhip"],  B["rbum"]],
    [B["lhip"],  B["rhip"]],  # hip width
    # === Thigh detail ===
    [B["lhip"],  B["lfrontthigh"]],
    [B["lhip"],  B["lfrontinnerthigh"]],
    [B["lkne"],  B["lthigh"]],
    [B["lkne"],  B["lknem"]],
    [B["lkne"],  B["linnerknee"]],
    [B["rhip"],  B["rfrontthigh"]],
    [B["rhip"],  B["rfrontinnerthigh"]],
    [B["rkne"],  B["rthigh"]],
    [B["rkne"],  B["rknem"]],
    [B["rkne"],  B["rinnerknee"]],
    # === Shin & ankle detail ===
    [B["lank"],  B["lankm"]],
    [B["lank"],  B["lshin"]],
    [B["rank"],  B["rankm"]],
    [B["rank"],  B["rshin"]],
    # === Foot detail ===
    [B["lank"],  B["lhee"]],
    [B["lhee"],  B["ltoe"]],
    [B["lank"],  B["lfifthmetatarsal"]],
    [B["lfoo"],  B["lfirstmetatarsal"]],
    [B["lfoo"],  B["lfourthtoe"]],
    [B["rank"],  B["rhee"]],
    [B["rhee"],  B["rtoe"]],
    [B["rank"],  B["rfifthmetatarsal"]],
    [B["rfoo"],  B["rfirstmetatarsal"]],
    [B["rfoo"],  B["rfourthtoe"]],
], dtype=int)

BML87_KEY_SUB = np.sort(np.unique(BML87_BONE.flatten()))
BML87_BONE_SUB = mk_bone_sub(BML87_BONE, BML87_KEY_SUB)


def get_bone_config(n_joints):
    """Return (bone_array, key_sub) based on the detected number of joints."""
    if n_joints == 87:
        return BML87_BONE, BML87_KEY_SUB
    if n_joints == 26:
        return METRABS_BONE, METRABS_KEY_SUB
    return OP_BONE, OP_KEY_SUB
