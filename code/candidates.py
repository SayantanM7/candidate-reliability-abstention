"""Stage 1 (candidate generation) and Stage 2 (HIGH/LOW reliability) for every case.

Stage 1: voxels with mean probability >= 0.10, 6-connected components; a component is kept
         whole if its maximum mean probability >= 0.80 (original component IDs are kept).
Stage 2: per candidate, the mean of the voxel variances; HIGH (1) if <= tau, otherwise LOW (2).

Usage: python code/candidates.py [--cases ...]   (default: all 100 cases)
"""
import argparse

import numpy as np
from scipy import ndimage as ndi

from common import case_file, gt_path, load_params, load_split, read, write


def stage1(mean_prob, fg=0.10, gate=0.80, connectivity=1):
    labels, n = ndi.label(mean_prob >= fg, structure=ndi.generate_binary_structure(3, connectivity))
    out = np.zeros(labels.shape, np.int32)
    if n:
        max_p = ndi.maximum(mean_prob, labels, index=np.arange(1, n + 1))
        for k in np.flatnonzero(np.asarray(max_p) >= gate) + 1:
            out[labels == k] = k
    return out


def stage2(labels, var_map, tau):
    """Returns the HIGH/LOW map and {candidate id: mean variance}."""
    out = np.zeros(labels.shape, np.uint8)
    mean_var = {}
    for k in np.unique(labels[labels > 0]):
        m = labels == k
        mean_var[int(k)] = float(var_map[m].mean())
        out[m] = 1 if mean_var[int(k)] <= tau else 2
    return out, mean_var


def candidate_rows(prm, case, split):
    """One row per Stage-1 candidate (columns of results/candidate_table_all.csv)."""
    labels = np.asarray(read(case_file(prm, case, "stage1"), np.float32)).astype(np.int32)
    conf = read(case_file(prm, case, "stage2"), np.float32)
    var = read(case_file(prm, case, "var_map"), np.float32)
    prob = read(case_file(prm, case, "mean_prob"), np.float32)
    gt = read(gt_path(prm, case)) > 0
    rows = []
    for k in np.unique(labels[labels > 0]):
        m = labels == k
        n, ov = int(m.sum()), int((m & gt).sum())
        rows.append(dict(case_id=case, split=split, blob_id=int(k), voxel_count=n, log_volume=float(np.log(n)),
                         mean_variance=float(var[m].mean()), max_prob=float(prob[m].max()),
                         gt_overlap=int(ov > 0), overlap_voxels=ov, purity=ov / n,
                         confidence="HIGH" if int(conf[m][0]) == 1 else "LOW"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()
    prm = load_params()
    train, val = load_split()
    s1, tau = prm["stage1"], prm["stage2"]["tau"]

    for case in args.cases or train + val:
        mp_path = case_file(prm, case, "mean_prob")
        labels = stage1(read(mp_path), s1["foreground_threshold"], s1["gate"], s1["connectivity"])
        conf, mean_var = stage2(labels, read(case_file(prm, case, "var_map"), np.float64), tau)
        write(labels, mp_path, case_file(prm, case, "stage1"), np.int32)
        write(conf, mp_path, case_file(prm, case, "stage2"), np.uint8)
        flags = ", ".join(f"{k}:{'HIGH' if v <= tau else 'LOW'} ({v:.5f})" for k, v in mean_var.items())
        print(f"{case}: {len(mean_var)} candidates  {flags}")


if __name__ == "__main__":
    main()
