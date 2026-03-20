from re import A
from util import *
import shutil
import glob
from scipy.stats import truncnorm

from argument import parse_args


def main(prefix, gt_dir, aid, pid, gid, trials, noise_level, th_truncate, cameras):

    if noise_level == 0:

        dst = f"{prefix}/noise_{noise_level}_{0}"
        shutil.copytree(gt_dir + "/2d_joint", dst + "/2d_joint", dirs_exist_ok=True)
        shutil.copytree(gt_dir + "/3d_joint", dst + "/3d_joint", dirs_exist_ok=True)
        shutil.copy(gt_dir + f"/cameras_G{gid:03d}.json", dst)
        shutil.copy(gt_dir + f"/skeleton_w_G{gid:03d}.json", dst)
    else:

        def gen_noises(x2d, sigma_xy):
            ## Generate noise
            noise_2d = truncnorm.rvs(
                -th_truncate * sigma_xy,
                th_truncate * sigma_xy,
                loc=0,
                scale=sigma_xy,
                size=(trials, 2),
            )
            x2d_noised = x2d + noise_2d

            return x2d_noised

        for t in range(trials):
            dst = f"{prefix}/noise_{noise_level}_{t}"
            shutil.copytree(gt_dir + "/2d_joint", dst + "/2d_joint", dirs_exist_ok=True)
            shutil.copy(gt_dir + f"/cameras_G{gid:03d}.json", dst)
            shutil.copy(gt_dir + f"/skeleton_w_G{gid:03d}.json", dst)

        _, x2d, _ = load_poses_all(gt_dir + "/2d_joint", cameras, aid, pid, gid)

        nC, nF, nJ, _ = x2d.shape

        x2ds_noised = []
        for x2d_i in x2d.reshape(-1, 2):
            x2ds_noised.append(gen_noises(x2d_i, noise_level))
        x2ds_noised = np.array(x2ds_noised)

        x2ds_noised = x2ds_noised.reshape(nC, nF, nJ, trials, 2)  ## CxFxJxTx2
        x2ds_noised = x2ds_noised.transpose(3, 0, 1, 2, 4)  ## TxCxFxJx2
        print("TxCxFxJx2")
        print(x2ds_noised.shape)

        ## Save
        TxCxFxJx2 = x2ds_noised
        for t in range(trials):
            for CxFxJx2 in TxCxFxJx2:
                dst = f"{prefix}/noise_{noise_level}_{t}"
                for filename, FxJx2 in zip(
                    sorted(glob.glob(os.path.join(dst, "2d_joint", "*.json"))), CxFxJx2
                ):

                    with open(filename, "r") as fp:
                        P2 = json.load(fp)
                    for (
                        frame2d,
                        Jx2,
                    ) in zip(P2["data"], FxJx2):
                        Jx2 = Jx2.flatten()
                        frame2d["skeleton"][0]["pose"] = Jx2.tolist()

                    with open(filename, "w") as fp:
                        json.dump(P2, fp, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    args = parse_args()
    PREFIX = args.prefix
    GT_DIR = args.prefix + "/gt_subset"

    AID = args.aid
    PID = args.pid
    GID = args.gid
    TRIALS = args.trials
    TH_TRUNCATE = args.th_truncate  ## truncate range to noise
    CAMERAS = [21, 23, 25, 27]

    NOISE_LEVEL = args.noise_level

    print(f"Noise Level: {NOISE_LEVEL}")

    main(PREFIX, GT_DIR, AID, PID, GID, TRIALS, NOISE_LEVEL, TH_TRUNCATE, CAMERAS)
