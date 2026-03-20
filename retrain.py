import os, sys

vp3d_path = os.path.abspath(os.path.join("./third_party/VideoPose3D"))
if vp3d_path not in sys.path:
    sys.path.append(vp3d_path)
import yaml
import matplotlib.pyplot as plt
from pycalib.calib import absolute_orientation
import traceback
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import errno
from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
import os
import sys
from util import select_gpu


N_JOINTS = 17
D_2D_JOINTS = 2


from argument import parse_args


def get_datetime_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Self_Supervision:
    def __init__(
        self, aid, pid, gid, retrain_pose, prefix, target, resume, retrain_target_epoch
    ):
        # self.args = args
        self.aid = aid
        self.pid = pid
        self.gid = gid
        self.subjects_train = ["S1", "S5", "S6", "S7", "S8"]
        self.subjects_test = ["S9", "S11"]
        self.disable_optimizations = False
        self.architecture = ["3", "3", "3", "3", "3"]
        self.stride = 1
        self.dropout = 0.25
        self.causal = True
        self.channels = 1024
        self.dense = False
        self.subset = 1
        self.downsample = 1
        self.lr_decay = 1.00
        self.bRetrain = True
        self.data_augmentation = True
        self.retrain_pose = retrain_pose
        self.retrain_ratio = 3
        self.data2d_path_ours = (
            prefix + "/" + target + "/" + "data_2d_gafa_detectron_pt_coco.npz"
        )
        self.data3d_path_ours = prefix + "/" + target + "/" + "data_3d_gafa.npz"
        self.checkpoint = "./model/"
        self.resume = resume

        self.kps_left, self.kps_right = [1, 3, 5, 7, 9, 11, 13, 15], [
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ]
        self.joints_left, self.joints_right = [4, 5, 6, 11, 12, 13], [
            1,
            2,
            3,
            14,
            15,
            16,
        ]

        self.keypoints_metadata = {
            "keypoints_symmetry": [self.kps_left, self.joints_right],
            "layout_name": "coco",
        }

        self.batch_size = 350
        self.learning_rate = 0.0005
        self.checkpoint_frequency = 1
        self.export_training_curves = True
        self.no_eval = False
        self.epochs = retrain_target_epoch

    def fetch(
        self,
        dataset,
        keypoints,
        subjects,
        action_filter=None,
        subset=1,
        parse_3d_poses=True,
    ):
        out_poses_3d = []  # _ []
        out_poses_2d = []  # _ []
        out_camera_params = []  # _ []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                # _ [(1360, 17, 2),(1360, 17, 2),(1360, 17, 2),(1360, 17, 2),]
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                if subject in dataset.cameras():
                    # _ [dict,dict,dict,dict,]
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), "Camera count mismatch"
                    for cam in cams:
                        if "intrinsic" in cam:
                            out_camera_params.append(cam["intrinsic"])
                        else:  # new
                            out_camera_params.append(None)

                if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                    # _ [(1360, 17, 3),(1360, 17, 3),(1360, 17, 3),(1360, 17, 3),]
                    poses_3d = dataset[subject][action]["positions_3d"]
                    assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                    for i in range(len(poses_3d)):  # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])

            if len(out_camera_params) == 0:
                out_camera_params = None
            if len(out_poses_3d) == 0:
                out_poses_3d = None

            stride = self.downsample  # _ int
            if subset < 1:
                for i in range(len(out_poses_2d)):
                    n_frames = int(
                        round(len(out_poses_2d[i]) // stride * subset) * stride
                    )
                    start = deterministic_random(
                        0,
                        len(out_poses_2d[i]) - n_frames + 1,
                        str(len(out_poses_2d[i])),
                    )
                    out_poses_2d[i] = out_poses_2d[i][start : start + n_frames : stride]
                    if out_poses_3d is not None:
                        out_poses_3d[i] = out_poses_3d[i][
                            start : start + n_frames : stride
                        ]
            elif stride > 1:
                # Downsample as requested
                for i in range(len(out_poses_2d)):
                    out_poses_2d[i] = out_poses_2d[i][::stride]
                    if out_poses_3d is not None:
                        out_poses_3d[i] = out_poses_3d[i][::stride]

            return out_camera_params, out_poses_3d, out_poses_2d

    def init_train(self, data_2d_h36m, data_3d_h36m):
        print("____________Loading h36m___________")
        dataset_main, keypoints_main = self.gen_dataset(
            "h36m", data_2d_h36m, data_3d_h36m
        )

        subjects_train_main = self.subjects_train
        subjects_test_main = self.subjects_test

        action_filter = None
        cameras_train_main, poses_train_main, poses_train_2d_main = self.fetch(
            dataset_main,
            keypoints_main,
            subjects_train_main,
            action_filter,
            subset=self.subset,
        )
        cameras_valid_main, poses_valid_main, poses_valid_2d_main = self.fetch(
            dataset_main, keypoints_main, subjects_test_main, action_filter
        )

        if self.bRetrain:
            print("____________Loading our data___________")
            dataset_ours, keypoints_ours = self.gen_dataset(
                "ours", self.data2d_path_ours, self.data3d_path_ours
            )

            print(self.retrain_pose)
            subjects_train_extra = [self.retrain_pose]
            subjects_test_extra = [self.retrain_pose]

            cameras_train_extra, poses_train_extra, poses_train_2d_extra = self.fetch(
                dataset_ours,
                keypoints_ours,
                subjects_train_extra,
                action_filter,
                subset=self.subset,
            )
            cameras_valid_extra, poses_valid_extra, poses_valid_2d_extra = self.fetch(
                dataset_ours, keypoints_ours, subjects_test_extra, action_filter
            )

            self.cameras_train = (
                cameras_train_main + cameras_train_extra * self.retrain_ratio
            )
            self.poses_train = poses_train_main + poses_train_extra * self.retrain_ratio
            self.poses_train_2d = (
                poses_train_2d_main + poses_train_2d_extra * self.retrain_ratio
            )
            self.keypoints = {
                **keypoints_main,
                **keypoints_ours,
                **keypoints_ours,
                **keypoints_ours,
            }

            self.cameras_valid = cameras_valid_main
            self.poses_valid = poses_valid_main
            self.poses_valid_2d = poses_valid_2d_main

        else:
            self.cameras_train = cameras_train_main
            self.poses_train = poses_train_main
            self.poses_train_2d = poses_train_2d_main
            self.keypoints = keypoints_main
            self.cameras_valid = cameras_valid_main
            self.poses_valid = poses_valid_main
            self.poses_valid_2d = poses_valid_2d_main

        filter_widths = [int(x) for x in self.architecture]
        if not self.disable_optimizations and not self.dense and self.stride == 1:
            # Use optimized model for single-frame predictions
            self.model_pos_train = TemporalModelOptimized1f(
                N_JOINTS,
                D_2D_JOINTS,
                N_JOINTS,
                filter_widths=filter_widths,
                causal=self.causal,
                dropout=self.dropout,
                channels=self.channels,
            )
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            self.model_pos_train = TemporalModel(
                N_JOINTS,
                D_2D_JOINTS,
                N_JOINTS,
                filter_widths=filter_widths,
                causal=self.causal,
                dropout=self.dropout,
                channels=self.channels,
                dense=self.dense,
            )

        self.model_pos = TemporalModel(
            N_JOINTS,
            D_2D_JOINTS,
            N_JOINTS,
            filter_widths=filter_widths,
            causal=self.causal,
            dropout=self.dropout,
            channels=self.channels,
            dense=self.dense,
        )

    def retrain(self):
        model_params = 0  # _ int
        for parameter in self.model_pos.parameters():
            model_params += parameter.numel()
        print("INFO: Trainable parameter count:", model_params)

        if torch.cuda.is_available():
            model_pos = self.model_pos.cuda()  # _ TemporalModel
            model_pos_train = self.model_pos_train.cuda()  # _ TemporalModel

        chk_filename = os.path.join(self.checkpoint, self.resume)  # _ str
        print("Loading checkpoint", chk_filename)
        checkpoint = torch.load(
            chk_filename, map_location=lambda storage, loc: storage
        )  # _ dict
        print("This model was trained for {} epochs".format(checkpoint["epoch"]))

        model_pos_train.load_state_dict(checkpoint["model_pos"])
        model_pos.load_state_dict(checkpoint["model_pos"])

        receptive_field = self.model_pos.receptive_field()  # _ int
        print("INFO: Receptive field: {} frames".format(receptive_field))
        pad = (receptive_field - 1) // 2  # Padding on each side  #_ int

        if self.causal:
            print("INFO: Using causal convolutions")
            causal_shift = pad  # _ int
        else:
            causal_shift = 0

        test_generator = UnchunkedGenerator(
            self.cameras_valid,
            self.poses_valid,
            self.poses_valid_2d,  # _ UnchunkedGenerator
            pad=pad,
            causal_shift=causal_shift,
            augment=False,
            kps_left=self.kps_left,
            kps_right=self.kps_right,
            joints_left=self.joints_left,  # _ []
            joints_right=self.joints_right,
        )  # _ []
        print("INFO: Testing on {} frames".format(test_generator.num_frames()))  # _ []

        lr = self.learning_rate  # _ float
        optimizer = optim.Adam(
            self.model_pos_train.parameters(), lr=lr, amsgrad=True
        )  # _ Adam

        lr_decay = self.lr_decay  # _ float

        losses_3d_train = []  # _ []
        losses_3d_train_eval = []  # _ []
        losses_3d_valid = []  # _ []

        epoch = 0  # _ int
        initial_momentum = 0.1  # _ float
        final_momentum = 0.001  # _ float
        # _ [dict,dict,dict,dict,]
        train_generator = ChunkedGenerator(
            self.batch_size // self.stride,
            self.cameras_train,
            self.poses_train,
            self.poses_train_2d,  # _ ChunkedGenerator
            self.stride,
            pad=pad,
            causal_shift=causal_shift,
            shuffle=True,
            augment=self.data_augmentation,
            kps_left=self.kps_left,
            kps_right=self.kps_right,
            joints_left=self.joints_left,
            joints_right=self.joints_right,
        )
        train_generator_eval = UnchunkedGenerator(
            self.cameras_train,
            self.poses_train,
            self.poses_train_2d,  # _ UnchunkedGenerator
            pad=pad,
            causal_shift=causal_shift,
            augment=False,
        )  # _ [(2499, 17, 3),(2499, 17, 3),(2499, 17, 3),(2499, 17, 3),]
        print("INFO: Training on {} frames".format(train_generator_eval.num_frames()))

        epoch = checkpoint["epoch"]  # _ int
        init_epoch = checkpoint["epoch"]
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            train_generator.set_random_state(checkpoint["random_state"])

        print(
            "** Note: reported losses are averaged over all frames and test-time augmentation is not used here."
        )
        print(
            "** The final evaluation will be carried out after the last training epoch."
        )

        epoch_x = []
        # Pos model only
        print("____________begin___________")
        print(f"------------------{epoch}")
        print(f"------------------{self.epochs}")
        while epoch < self.epochs:
            print(f"_______epoch{epoch}________")
            start_time = time()  # _ float
            epoch_loss_3d_train = 0  # _ int
            N = 0  # _ int

            self.model_pos_train.train()
            # Regular supervised scenario
            for _, batch_3d, batch_2d in train_generator.next_epoch():
                inputs_3d = torch.from_numpy(
                    batch_3d.astype("float32")
                )  # _ torch.Size([350, 1, 17, 3])
                # _ torch.Size([350, 243, 17, 2])
                inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                if torch.cuda.is_available():
                    # _ torch.Size([350, 1, 17, 3])
                    inputs_3d = inputs_3d.cuda()
                    # _ torch.Size([350, 243, 17, 2])
                    inputs_2d = inputs_2d.cuda()
                inputs_3d[:, :, 0] = 0  # _ int

                optimizer.zero_grad()

                # Predict 3D poses
                predicted_3d_pos = self.model_pos_train(
                    inputs_2d
                )  # _ torch.Size([350, 1, 17, 3])
                # _ torch.Size([])
                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_train += (
                    inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                )
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                loss_total = loss_3d_pos  # _ torch.Size([])
                loss_total.backward()

                optimizer.step()
            #                 print("FULLY SUPERVISED TERMINATE")
            #             break

            losses_3d_train.append(epoch_loss_3d_train / N)

            # End-of-epoch evaluation
            with torch.no_grad():
                self.model_pos.load_state_dict(self.model_pos_train.state_dict())
                self.model_pos.eval()

                epoch_loss_3d_valid = 0  # _ int
                N = 0  # _ int

                if not self.no_eval:
                    # Evaluate on test set
                    for _, batch, batch_2d in test_generator.next_epoch():
                        # _ torch.Size([1, 2356, 17, 3])
                        inputs_3d = torch.from_numpy(batch.astype("float32"))
                        # _ torch.Size([1, 2598, 17, 2])
                        inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                        if torch.cuda.is_available():
                            # _ torch.Size([1, 1360, 17, 3])
                            inputs_3d = inputs_3d.cuda()
                            # _ torch.Size([1, 1602, 17, 2])
                            inputs_2d = inputs_2d.cuda()
                        # _ torch.Size([1, 1360, 1, 3])

                        inputs_3d[:, :, 0] = 0  # _ int

                        # Predict 3D poses
                        # _ torch.Size([1, 1360, 17, 3])
                        predicted_3d_pos = self.model_pos(inputs_2d)
                        # _ torch.Size([])
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                        epoch_loss_3d_valid += (
                            inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                        )
                        N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    losses_3d_valid.append(epoch_loss_3d_valid / N)

                    # Evaluate on training set, this time in evaluation mode
                    epoch_loss_3d_train_eval = 0  # _ int

                    N = 0  # _ int
                    for _, batch, batch_2d in train_generator_eval.next_epoch():
                        if batch_2d.shape[1] == 0:
                            # This can only happen when downsampling the dataset
                            continue

                        # _ torch.Size([1, 2499, 17, 3])
                        inputs_3d = torch.from_numpy(batch.astype("float32"))
                        # _ torch.Size([1, 2741, 17, 2])
                        inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                        if torch.cuda.is_available():
                            # _ torch.Size([1, 2499, 17, 3])
                            inputs_3d = inputs_3d.cuda()
                            # _ torch.Size([1, 2741, 17, 2])
                            inputs_2d = inputs_2d.cuda()
                        # _ torch.Size([1, 2499, 1, 3])
                        inputs_3d[:, :, 0] = 0  # _ int

                        # Compute 3D poses
                        # _ torch.Size([1, 2499, 17, 3])
                        predicted_3d_pos = self.model_pos(inputs_2d)
                        # _ torch.Size([])
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                        epoch_loss_3d_train_eval += (
                            inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                        )
                        N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

            elapsed = (time() - start_time) / 60  # _ float

            if self.no_eval:
                print(
                    "[%d] time %.2f lr %f 3d_train %f"
                    % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000)
                )
            else:
                print(
                    "[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f"
                    % (
                        epoch + 1,
                        elapsed,
                        lr,
                        losses_3d_train[-1] * 1000,
                        losses_3d_train_eval[-1] * 1000,
                        losses_3d_valid[-1] * 1000,
                    )
                )

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay
            epoch += 1

            # Decay BatchNorm momentum
            momentum = initial_momentum * np.exp(
                -epoch / self.epochs * np.log(initial_momentum / final_momentum)
            )  # _ ()
            self.model_pos_train.set_bn_momentum(momentum)

            # Save checkpoint if necessary
            if epoch % self.checkpoint_frequency == 0:
                chk_path = os.path.join(
                    self.checkpoint, "epoch_{}.bin".format(epoch)
                )  # _ str
                print("Saving checkpoint to", chk_path)

                torch.save(
                    {
                        "epoch": epoch,
                        "lr": lr,
                        "random_state": train_generator.random_state(),
                        "optimizer": optimizer.state_dict(),
                        "model_pos": self.model_pos_train.state_dict(),
                        "model_traj": None,
                        "random_state_semi": None,
                    },
                    chk_path,
                )

            # Save training curves after every epoch, as .png images (if requested)
            if self.export_training_curves and epoch > 3:

                plt.figure()
                # epoch_x = np.arange(3, len(losses_3d_train)) + 1  # _ (0,)
                epoch_x.append(epoch)
                # print(True, losses_3d_train,
                #         losses_3d_train_eval, losses_3d_valid)
                plt.plot(epoch_x, losses_3d_train, "--", color="C0")
                plt.plot(epoch_x, losses_3d_train_eval, color="C0")
                plt.plot(epoch_x, losses_3d_valid, color="C1")
                plt.legend(["3d train", "3d train (eval)", "3d valid (eval)"])
                plt.ylabel("MPJPE (m)")
                plt.xlabel("Epoch")
                plt.xlim((init_epoch + 1, self.epochs))
                plt.savefig(os.path.join(self.checkpoint, f"loss_3d_{self.epochs}.png"))
                plt.close("all")

    def prep_3d_data(self, dataset):

        print("Preparing data...")
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                anim = dataset[subject][action]  # _ dict
                if "positions" in anim:
                    positions_3d = []  # _ []
                    for cam in anim["cameras"]:
                        pos_3d = world_to_camera(
                            anim["positions"],
                            R=cam["orientation"],
                            t=cam["translation"],
                        )  # _ (1360, 17, 3)
                        # print(pos_3d[0])

                        # Remove global offset, but keep trajectory in first position
                        pos_3d[:, 1:] -= pos_3d[:, :1]
                        # print(pos_3d[0])
                        # print('-----')
                        positions_3d.append(pos_3d)

                    # _ [(1360, 17, 3),(1360, 17, 3),(1360, 17, 3),(1360, 17, 3),]
                    anim["positions_3d"] = positions_3d

        return dataset

    def gen_dataset(self, dataset_name, dataset_2d_path, dataset_3d_path):

        print("Loading dataset...")

        if dataset_name == "h36m":
            from common.h36m_dataset import Human36mDataset

            dataset = Human36mDataset(dataset_3d_path)
            dataset = self.prep_3d_data(dataset)

            keypoints = np.load(dataset_2d_path, allow_pickle=True)
            keypoints = keypoints["positions_2d"].item()

        elif dataset_name == "ours":

            from my_dataset import MyDataset

            dataset = MyDataset(dataset_3d_path, dataset_2d_path)
            dataset = self.prep_3d_data(dataset)
            keypoints = dataset.keypoints
        else:
            raise KeyError("Invalid dataset")

        for subject in dataset.subjects():
            assert (
                subject in keypoints
            ), "Subject {} is missing from the 2D detections dataset".format(subject)
            for action in dataset[subject].keys():
                assert (
                    action in keypoints[subject]
                ), "Action {} of subject {} is missing from the 2D detections dataset".format(
                    action, subject
                )
                if "positions_3d" not in dataset[subject][action]:
                    continue

                for cam_idx in range(len(keypoints[subject][action])):

                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    # _ int
                    mocap_length = dataset[subject][action]["positions_3d"][
                        cam_idx
                    ].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        # _ (1360, 17, 2)
                        keypoints[subject][action][cam_idx] = keypoints[subject][
                            action
                        ][cam_idx][:mocap_length]

                assert len(keypoints[subject][action]) == len(
                    dataset[subject][action]["positions_3d"]
                )

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    # Normalize camera frame
                    cam = dataset.cameras()[subject][cam_idx]  # _ dict

                    if dataset_name == "h36m":
                        kps = normalize_screen_coordinates(
                            kps[..., :2], w=cam["res_w"], h=cam["res_h"]
                        )  # _ (2340, 17, 2)
                    else:
                        kps[..., :2] = normalize_screen_coordinates(
                            kps[..., :2], w=cam["res_w"], h=cam["res_h"]
                        )
                    # _ (2340, 17, 2)
                    keypoints[subject][action][cam_idx] = kps

        return dataset, keypoints


def main(aid, pid, gid, retrain_pose, prefix, target, resume, retrain_target_epoch):

    retrain = Self_Supervision(
        aid, pid, gid, retrain_pose, prefix, target, resume, retrain_target_epoch
    )
    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())
    data_2d_h36m = config["RETRAIN"]["data_2d_h36m"]
    data_3d_h36m = config["RETRAIN"]["data_3d_h36m"]
    retrain.init_train(data_2d_h36m, data_3d_h36m)
    retrain.retrain()


if __name__ == "__main__":

    args = parse_args()
    select_gpu()
    PREFIX = args.prefix + "/" + args.target
    AID = args.aid
    PID = args.pid
    GID = args.gid

    retrain_pose = args.retrain_pose

    main(
        AID,
        PID,
        GID,
        retrain_pose,
        args.prefix,
        args.target,
        args.retrain_resume,
        args.retrain_target_epoch,
    )

    # Train = TrainVP3D("h36m", data_3d_h36m, data_2d_h36m, retraining=True)
