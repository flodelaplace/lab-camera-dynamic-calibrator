import argparse

# from xml.etree.ElementInclude import default_loader
from distutils.util import strtobool


def parse_args(predefined_args=None):

    parser = argparse.ArgumentParser(description="human_calib")
    parser.add_argument("--dataset", type=str, default="SynADL")
    parser.add_argument("--aid", type=int, default=23)
    parser.add_argument("--pid", type=int, default=102)
    parser.add_argument("--gid", type=int, default=3)
    parser.add_argument("--cid", type=int, default=21)
    parser.add_argument("--calib", nargs="+", type=int, default=[0])
    parser.add_argument("--frame_skip", type=int, default=15)
    parser.add_argument("--noise_level", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="./third_party/SynADL/")

    parser.add_argument("--ransac_th_2d", type=float, default="50")
    parser.add_argument("--ransac_th_3d", type=float, default="5")
    parser.add_argument("--ransac_niter", type=int, default="100")
    parser.add_argument("--ransac_seed", type=int, default="0")
    parser.add_argument("--target", type=str, default="noise_3_0")

    parser.add_argument("--th_truncate", type=float, default=10)
    parser.add_argument("--ba_lambda1", type=float, default=1.0)
    parser.add_argument("--ba_lambda2", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=15)

    parser.add_argument(
        "--model", type=str, default="pretrained_h36m_detectron_coco.bin"
    )

    parser.add_argument("--retrain_pose", type=str, default="linear_77_0_ba")
    parser.add_argument(
        "--retrain_resume", type=str, default="pretrained_h36m_detectron_coco.bin"
    )
    parser.add_argument("--retrain_target_epoch", type=int, default=100)

    parser.add_argument("--obs_mask", type=strtobool, default=False)
    parser.add_argument("--save_obs_mask", type=strtobool, default=False)
    parser.add_argument("--th_obs_mask", type=int, default=20)

    parser.add_argument(
        "--src_original", type=str, default="./third_party/GAFA/lab/1013_2"
    )

    parser.add_argument("--vis_type", type=str, default="2d")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu)")
    parser.add_argument("--pose_engine", type=str, default="rtmpose", choices=["rtmpose", "metrabs"], help="Pose estimation engine: rtmpose (2D+lifting) or metrabs (direct 3D)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for 2D keypoints")
    
    # Arguments for chunking
    parser.add_argument("--frame_start", type=int, default=None, help="Start frame for chunk processing.")
    parser.add_argument("--frame_end", type=int, default=None, help="End frame for chunk processing.")
    parser.add_argument("--chunk_id", type=int, default=None, help="Identifier for the current chunk.")


    if predefined_args == None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=predefined_args)

    ## For executing in ipynb

    return args
