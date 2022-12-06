from adorym.ptychography import reconstruct_ptychography
import adorym
import numpy as np
import dxchange
import datetime
import argparse
import os

output_folder = "SiemensLH_holonfp_15nm_test0"
distribution_mode = None
optimizer_obj = adorym.AdamOptimizer(
    "obj",
    output_folder=output_folder,
    distribution_mode=distribution_mode,
    options_dict={"step_size": 1e-3},
)
optimizer_probe = adorym.AdamOptimizer(
    "probe",
    output_folder=output_folder,
    distribution_mode=distribution_mode,
    options_dict={"step_size": 1e-3, "eps": 1e-7},
)
optimizer_all_probe_pos = adorym.AdamOptimizer(
    "probe_pos_correction",
    output_folder=output_folder,
    distribution_mode=distribution_mode,
    options_dict={"step_size": 1e-2},
)

params_2idd_gpu = {
    "fname": "SiemensLH_holonfp_15nm.h5",
    "theta_st": 0,
    "theta_end": 0,
    "n_epochs": 1000,
    "obj_size": (2048, 2048, 1),
    "two_d_mode": True,
    "minibatch_size": 35,
    "output_folder": output_folder,
    "cpu_only": False,
    "save_path": "/nrs/funke/rhoadesj/data/XNH/Adorym/",
    "use_checkpoint": False,
    "n_epoch_final_pass": None,
    "save_intermediate": True,
    "full_intermediate": True,
    "initial_guess": None,
    "random_guess_means_sigmas": (1.0, 0.0, 0.001, 0.002),
    "n_dp_batch": 350,
    # ===============================
    "probe_type": "ifft",
    "n_probe_modes": 5,
    # ===============================
    # "rescale_probe_intensity": True,
    # "free_prop_cm": "inf",
    "backend": "pytorch",
    "raw_data_type": "intensity",
    "beamstop": None,
    "optimizer": optimizer_obj,
    "optimize_probe": True,
    "optimizer_probe": optimizer_probe,
    "optimize_all_probe_pos": True,
    "optimizer_all_probe_pos": optimizer_all_probe_pos,
    "save_history": True,
    "update_scheme": "immediate",
    "unknown_type": "real_imag",
    "save_stdout": True,
    "loss_function_type": "lsq",
    # "normalize_fft": False,
    # =============================== Jeff added
    "optimize_probe_defocusing": True,
    "optimize_free_prop": True,
    "optimize_prj_affine": True,
    "optimize_prj_pos_offset": True,
    "multiscale_level": 4,
    "randomize_probe_pos": True,
    "forward_model": adorym.forward_model.MultiDistModel,
}

params = params_2idd_gpu

reconstruct_ptychography(**params)
