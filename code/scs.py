"""Spatial Concern Score (SCS) for HIGH candidates, and the portal-vein distance map it uses.

  U_N = U / (U + U0)                                   normalised local variance, U0 = 0.05
  P   = 1 if D <= 5 mm, else exp(-(D - 5) / 5)         portal-vein proximity
  R   = 1 + 6.5 P + 4 U_N - 2 P U_N, and R = 10 if D <= 5 mm (maximum-concern constraint)

The coefficients reproduce the radiologist's four HIGH-candidate ratings (sharp/far 1,
sharp/near 7.5, fuzzy/far 5, fuzzy/near 9.5). The SCS is a prioritisation measure, not a
clinically validated risk score; LOW candidates receive no score.

Usage: python code/scs.py [--cases ...]   writes portal_distance_mm.nii.gz for each case
"""
import argparse
import math

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from common import case_file, ct_path, load_params, load_split, write

PRM = load_params()["scs"]


def normalised_uncertainty(u, u0=PRM["u0"]):
    return u / (u + u0)


def proximity(d_mm, d_ref=PRM["d_ref_mm"], sigma=PRM["sigma_mm"]):
    return 1.0 if d_mm <= d_ref else math.exp(-(d_mm - d_ref) / sigma)


def spatial_concern_score(u, d_mm):
    """SCS at one voxel from its local variance u and portal-vein distance d_mm."""
    if d_mm <= PRM["d_ref_mm"]:
        return PRM["max_score"]
    c0, c1, c2, c3 = PRM["coefficients"]
    un, p = normalised_uncertainty(u), proximity(d_mm)
    return c0 + c1 * p + c2 * un + c3 * p * un


def portal_distance_map(mask_path, reference_ct):
    """Euclidean distance (mm) from every voxel to the nearest portal-vein voxel (0 inside)."""
    mask_img, ct = nib.load(str(mask_path)), nib.load(str(reference_ct))
    if mask_img.shape[:3] != ct.shape[:3] or not np.allclose(mask_img.affine, ct.affine, atol=1e-3):
        raise ValueError(f"{mask_path}: portal-vein mask is not on the CT grid")
    vein = mask_img.get_fdata() > 0
    if not vein.any():
        raise ValueError(f"{mask_path}: empty portal-vein mask")
    return distance_transform_edt(~vein, sampling=mask_img.header.get_zooms()[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()
    prm = load_params()
    train, val = load_split()
    for case in args.cases or train + val:
        d = portal_distance_map(case_file(prm, case, "portal_vein_mask"), ct_path(prm, case))
        write(d, ct_path(prm, case), case_file(prm, case, "portal_distance"), np.float32)
        print(f"{case}: distance map {d.min():.1f}-{d.max():.1f} mm")


if __name__ == "__main__":
    main()
