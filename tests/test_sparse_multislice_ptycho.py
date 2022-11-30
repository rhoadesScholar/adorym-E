from adorym.ptychography import reconstruct_ptychography
import numpy as np
import dxchange
import datetime
import argparse
import os

timestr = str(datetime.datetime.today())
timestr = timestr[: timestr.find(".")]
for i in [":", "-", " "]:
    if i == " ":
        timestr = timestr.replace(i, "_")
    else:
        timestr = timestr.replace(i, "")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default="None")
parser.add_argument("--save_path", default="cone_256_foam_ptycho")
parser.add_argument(
    "--output_folder", default="test"
)  # Will create epoch folders under this
args = parser.parse_args()
epoch = args.epoch
if epoch == "None":
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(
            os.path.join(
                args.save_path,
                args.output_folder,
                "epoch_{}/delta_ds_1.tiff".format(epoch - 1),
            )
        )
        init_beta = dxchange.read_tiff(
            os.path.join(
                args.save_path,
                args.output_folder,
                "epoch_{}/beta_ds_1.tiff".format(epoch - 1),
            )
        )
        print(
            os.path.join(
                args.save_path,
                args.output_folder,
                "epoch_{}/delta_ds_1.tiff".format(epoch - 1),
            )
        )
        init = [np.array(init_delta[...]), np.array(init_beta[...])]

params_2idd_gpu = {
    "fname": "data.h5",
    "theta_st": 0,
    "theta_end": 0,
    "theta_downsample": None,
    "n_epochs": 2000,
    "obj_size": (501, 500, 2),
    # 'obj_size': (50, 50, 1),
    "alpha_d": 0,
    "alpha_b": 0,
    "gamma": 0,
    "two_d_mode": True,
    "learning_rate": 1e-3,
    "probe_learning_rate": 1e-3,
    "minibatch_size": 202,
    "randomize_probe_pos": False,
    "n_batch_per_update": 1,
    #'output_folder': 'rec_ol1e-3_nslice2_nmode1_init_mask',
    "output_folder": "test",
    "cpu_only": False,
    "save_path": "../demos/sparse_multislice_ptycho",
    "multiscale_level": 1,
    "use_checkpoint": False,
    "store_checkpoint": False,
    "n_epoch_final_pass": None,
    "save_intermediate": True,
    "full_intermediate": True,
    "initial_guess": None,
    "random_guess_means_sigmas": (1.0, 0.0, 0.001, 0.002),
    "n_dp_batch": 350,
    "optimize_probe": True,
    "probe_update_delay": 50,
    "slice_pos_cm_ls": [0, 10e-4],
    # ===============================
    "probe_type": "ifft",
    "probe_extra_defocus_cm": 20e-4,
    #'probe_mag_sigma': 12,
    #'probe_phase_sigma': 12,
    #'probe_phase_max': 0.3,
    # ===============================
    "probe_initial": None,
    "rescale_probe_intensity": True,
    "forward_algorithm": "fresnel",
    "finite_support_mask_path": None,
    "shared_file_object": False,
    "reweighted_l1": False,
    "optimizer": "adam",
    "free_prop_cm": "inf",
    "backend": "pytorch",
    "raw_data_type": "magnitude",
    "beamstop": None,
    "debug": False,
    "optimize_all_probe_pos": False,
    "all_probe_pos_learning_rate": 1e-2,
    "optimize_slice_pos": False,
    "slice_pos_learning_rate": 1e-5,
    "save_history": True,
    "update_scheme": "immediate",
    "unknown_type": "real_imag",
    "save_stdout": True,
    "loss_function_type": "lsq",
    "normalize_fft": False,
    "sign_convention": 1,
}

params = params_2idd_gpu

reconstruct_ptychography(**params)
