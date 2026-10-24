"""Shared helpers: parameters, split, per-case paths and NIfTI I/O."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_params(path=REPO / "config" / "parameters.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(path=REPO / "config" / "split.json"):
    with open(path) as f:
        fold = json.load(f)[0]
    return sorted(fold["train"]), sorted(fold["val"])


def case_file(prm, case, key):
    return Path(prm["paths"]["work_dir"]) / case / prm["files"][key]


def ct_path(prm, case):
    return Path(prm["paths"]["nnunet_raw"]) / "imagesTr" / f"{case}_0000.nii.gz"


def gt_path(prm, case):
    return Path(prm["paths"]["nnunet_raw"]) / "labelsTr" / f"{case}.nii.gz"


def read(path, dtype=np.float32):
    """Read a NIfTI volume with any NIfTI scaling applied."""
    return nib.load(str(path)).get_fdata(dtype=dtype)


def write(array, reference_path, out_path, dtype):
    """Save `array` with the geometry of the reference image."""
    ref = nib.load(str(reference_path))
    if array.shape != ref.shape[:3]:
        raise ValueError(f"{out_path}: shape {array.shape} != reference {ref.shape[:3]}")
    header = ref.header.copy()
    header.set_data_dtype(dtype)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(dtype), ref.affine, header), str(out_path))
