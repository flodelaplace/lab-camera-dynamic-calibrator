from pickle import FALSE, TRUE
import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset

# from common.camera import normalize_screen_coordinates, image_coordinates
from common.h36m_dataset import h36m_skeleton
from common.skeleton import Skeleton
import copy


class MyDataset(MocapDataset):
    def __init__(self, dataset_3d_path, dataset_2d_path, remove_static_joints=True):
        super().__init__(fps=30, skeleton=copy.deepcopy(h36m_skeleton))

        loaded_data = np.load(dataset_3d_path, allow_pickle=True)

        data = loaded_data["positions_3d"].item()
        self._cameras_ori = loaded_data["cameras"].item()

        print("Loading 2D detections...")

        keypoints = np.load(dataset_2d_path, allow_pickle=True)

        self._keypoints_ori = keypoints["positions_2d"].item()

        self.keypoints = {}

        for scene_name, custom in self._keypoints_ori.items():

            newcustom = {}
            kpts = []
            for cam_id, kpt in custom["custom"].items():

                kpts.append(kpt)

            newcustom["custom"] = kpts
            self.keypoints[scene_name] = newcustom

        self._cameras = {}

        for scene_name, camid_by in self._cameras_ori.items():
            cams_info = []
            for cam_id, caminfo in camid_by.items():
                cams_info.append(caminfo)
            self._cameras[scene_name] = cams_info

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    "positions": positions,
                    "cameras": self._cameras[subject],
                }
        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints(
                [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
            )
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return False
