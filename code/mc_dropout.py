"""MC-Dropout inference: 30 stochastic passes -> mean_prob.nii.gz and var_map.nii.gz per case.

Mean and population variance (ddof = 0) of the tumour-channel probability over the passes,
transposed from nnU-Net's (z, y, x) to the CT's (x, y, z) and saved with the CT geometry.
The paper's maps were written with the CT header (int16 with NIfTI scaling); float32 here
differs by ~1e-6 and changes no candidate label.

Usage: python code/mc_dropout.py [--cases HCC_045 HCC_077 ...]   (default: all 100 cases)
"""
import argparse

import numpy as np
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from common import case_file, ct_path, load_params, load_split, write


def build_predictor(prm):
    inf = prm["inference"]
    predictor = nnUNetPredictor(tile_step_size=inf["tile_step_size"], use_gaussian=inf["use_gaussian"],
                                use_mirroring=inf["use_mirroring"], perform_everything_on_device=True,
                                device=torch.device("cuda"))
    predictor.initialize_from_trained_model_folder(prm["paths"]["model_folder"], use_folds=(inf["fold"],),
                                                   checkpoint_name=inf["checkpoint"])
    predictor.network.eval()

    def keep_dropout_active(module, inputs, output):          # dropout stays on at inference
        return torch.nn.functional.dropout3d(inputs[0], p=module.p, training=True)

    n = 0
    for m in predictor.network.modules():
        if isinstance(m, torch.nn.Dropout3d):
            m.register_forward_hook(keep_dropout_active)
            n += 1
    if n != 22:
        raise RuntimeError(f"expected 22 Dropout3d layers, found {n}")
    return predictor


def mc_maps(predictor, image, props, n_passes):
    """Mean and population variance of the tumour probability over n_passes, in (z, y, x)."""
    s = s2 = None
    for i in range(n_passes):
        _, probs = predictor.predict_single_npy_array(image, props, None, None, True)
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        p = np.asarray(probs[1], dtype=np.float64)              # tumour channel
        s, s2 = (p, p * p) if s is None else (s + p, s2 + p * p)
        print(f"  pass {i + 1}/{n_passes}", end="\r")
    mean = s / n_passes
    return mean, np.maximum(s2 / n_passes - mean * mean, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()
    prm = load_params()
    train, val = load_split()
    cases = args.cases or train + val

    predictor = build_predictor(prm)
    reader = SimpleITKIO()
    for case in cases:
        print(case)
        image, props = reader.read_images([str(ct_path(prm, case))])
        mean, var = mc_maps(predictor, image, props, prm["inference"]["n_passes"])
        for arr, key in ((mean, "mean_prob"), (var, "var_map")):    # (z, y, x) -> (x, y, z)
            write(np.transpose(arr, (2, 1, 0)), ct_path(prm, case), case_file(prm, case, key), np.float32)


if __name__ == "__main__":
    main()
