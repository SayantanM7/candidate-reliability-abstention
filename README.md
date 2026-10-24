# Knowing When Not to Trust: Uncertainty-Based Abstention for Hepatocellular Carcinoma Candidates in CT

Code and results for a candidate-level reliability flag for automatically generated
hepatocellular carcinoma (HCC) candidates in contrast-enhanced CT, based on Monte Carlo (MC) Dropout.

**Authors:** Sayantan Mandal (Independent Researcher, Kolkata, India and ex-student of VNIT Nagpur, Maharashtra, India) and
Ashish Tiwari (VNIT Nagpur, India).
The accompanying paper is currently under review.

> Research prototype. HIGH/LOW denotes **model reliability, not HCC probability**.
> The framework does not diagnose HCC or select treatment.

## Overview

Aggregate segmentation accuracy does not show whether an individual predicted tumour can be trusted.
This framework asks, for each automatically generated candidate, whether it is reliable enough for
downstream interpretation, and abstains when it is not.

1. **Candidate generation (Stage 1).** An nnU-Net with Dropout3d produces mean probability and
   variance maps from 30 stochastic passes; candidates are connected components of the mean map.
2. **Reliability (Stage 2).** Each candidate is labelled **HIGH** or **LOW** by its mean variance.
   LOW candidates are an **abstention**: they stay visible for review and are not treated as non-tumour.
3. **Spatial concern (HIGH only).** An expert-anchored Spatial Concern Score (SCS) combines local
   uncertainty with portal-vein proximity. LOW candidates receive no score.
4. **Interpretable prototype.** A Tkinter interface reports, for a queried voxel, the reliability
   label, local variance, portal-vein distance and, for HIGH candidates, the SCS with an explanation.

## Method

| Step | Setting |
|---|---|
| Data | HCC-TACE-Seg, 100 cases, binary tumour label; nnU-Net fold 0: 80 development, 20 validation cases |
| Network | nnU-Net v2, 3D full resolution, 128³ patches, 2.5 × 0.78 × 0.78 mm |
| Training | Pretrained on LiTS (binary tumour label, no dropout); fine-tuned on the 80 development cases with Dropout3d (p = 0.2), configuration `3d_fullres_mcdropout`; checkpoint epoch 295 |
| Inference | 30 passes with all 22 Dropout3d layers active, mirroring and Gaussian blending; mean probability p̄ and population variance U |
| Stage 1 | p̄ ≥ 0.10, 6-connected components, keep a component if max p̄ ≥ 0.80 |
| Stage 2 | HIGH if the candidate's mean variance ≤ τ = 0.01929155, otherwise LOW; τ maximised F1 for tumour overlap over the 132 in-sample development candidates and was frozen before validation |
| SCS (HIGH only) | U_N = U / (U + 0.05); P = 1 for D ≤ 5 mm, else exp(−(D − 5)/5); R = 1 + 6.5P + 4U_N − 2PU_N, and R = 10 for D ≤ 5 mm |

D is the 3D distance (mm) from the queried voxel to the expert portal-vein annotation provided with
HCC-TACE-Seg (not predicted by the network). The SCS coefficients reproduce four concern ratings given
by an interventional radiologist; the score is an interpretable prioritisation measure, **not a
clinically validated risk or safety score**.

## Results (20 validation cases, 31 candidates)

| Measure | Result |
|---|---|
| HIGH candidates overlapping tumour | 14 / 14 (100%; exact 95% CI 76.8–100%) |
| LOW candidates overlapping tumour | 4 / 17 (23.5%) |
| Non-overlapping candidates labelled LOW | 13 / 13 |
| Local Dice with the reference lesion, median | HIGH 0.82 (n = 14) vs LOW 0.62 (n = 4); exact Mann–Whitney p = 0.008 (exploratory) |
| Mean Dice in the 14 cases with a HIGH candidate | 0.738 (raw) → 0.746 (Stage 1) → 0.801 (Stage 2) |
| Candidate-overlap AUC | variance 0.876 vs maximum probability 0.846 (paired case-level bootstrap of the difference: +0.029, 95% CI −0.064 to 0.118) |

The flag trades coverage for reliability: across all 20 cases, mean Dice is 0.616 (raw), 0.631
(Stage 1) and 0.561 (Stage 2), because the six cases without a HIGH candidate are deferred.
Variance is treated as a complementary reliability signal, not a superior detector.
Results are preliminary (small internal cohort); external validation is future work.

## Repository structure

```
uncertainty-aware-hcc/
├── README.md
├── LICENSE.txt
├── requirements.txt
├── config/
│   ├── split.json            # nnU-Net fold-0 split (80 development / 20 validation)
│   ├── nnunet_plans.json     # plans incl. 3d_fullres_mcdropout (Dropout3d, p = 0.2)
│   └── parameters.yaml       # all thresholds and SCS constants used in the paper
├── training/
│   ├── README.md             # exact LiTS pretraining and HCC fine-tuning commands
│   └── prepare_fine_tuning.py
├── code/
│   ├── common.py             # parameters, split, paths, NIfTI I/O
│   ├── mc_dropout.py         # 30 passes -> mean_prob.nii.gz, var_map.nii.gz
│   ├── candidates.py         # Stage 1 and Stage 2
│   ├── calibrate_tau.py      # F1-maximising threshold on development candidates
│   ├── evaluate.py           # Table 1, local Dice, Mann–Whitney, AUCs, paired bootstrap
│   └── scs.py                # portal-vein distance map, U_N, P(D), SCS
├── prototype/
│   └── gui.py                # Tkinter interface
└── results/
    ├── candidate_table_all.csv   # all 163 candidates (132 development, 31 validation)
    └── pair_table.csv            # validation lesion–candidate pairs (local Dice)
```

## Usage

1. **Data.** Download HCC-TACE-Seg from The Cancer Imaging Archive
   (doi:10.7937/TCIA.5FNA-0924) and LiTS. Images and masks are not redistributed here.
2. **Training.** Follow `training/README.md` (LiTS pretraining, plan alignment, fine-tuning with
   `3d_fullres_mcdropout`).
3. **Inference.** `python code/mc_dropout.py` writes `mean_prob.nii.gz` and `var_map.nii.gz` per case.
4. **Candidates and reliability.** `python code/candidates.py` writes Stage-1 and Stage-2 label maps;
   `python code/calibrate_tau.py` reproduces τ from the development candidates.
5. **Evaluation.** `python code/evaluate.py` reproduces the validation results above
   (`--from-csv` recomputes Table 1, local Dice and the AUCs from `results/` without the images).
6. **Spatial concern.** Place each case's portal-vein annotation in its work folder as `portal_vein_mask.nii.gz`;
   `python code/scs.py` writes the distance maps.
7. **Prototype.** `python prototype/gui.py`; enter a case ID and a voxel index (i, j, k) read from 3D Slicer.

All thresholds and constants are read from `config/parameters.yaml`.
Inference was run with nnU-Net v2 (nnunetv2 2.8.1); see `requirements.txt`.

## Citation

If you use this code, please cite the accompanying paper (citation details will be added after review).

## Acknowledgements

We thank an interventional radiologist for the expert ratings used to anchor the spatial concern
score, and a medical oncologist for clinical guidance.

## License

This project is released under the [MIT License](LICENSE.txt).
