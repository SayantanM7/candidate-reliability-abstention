# candidate-reliability-abstention
Uncertainty-aware candidate reliability using stochastic prediction, selective abstention, and interpretable spatial assessment.

# Uncertainty-Aware Candidate Reliability

Code and research materials for uncertainty-aware candidate reliability
using stochastic prediction, selective abstention, and interpretable
spatial assessment.

## Overview

This repository accompanies the paper:

> **Uncertainty-Aware Candidate Reliability and Explainable Spatial
> Concern Assessment for HCC Tumor Detection in CT**

The framework uses Monte Carlo Dropout to estimate predictive
uncertainty for automatically generated candidates. Candidates are
then stratified into HIGH- and LOW-reliability groups using candidate-
level predictive variance.

HIGH-reliability candidates are retained for downstream spatial
assessment, while LOW-reliability candidates are explicitly abstained
and flagged for review rather than being treated as negative
predictions.

The framework consists of three main stages:

1. **Candidate generation**  
   Candidate regions are extracted from the mean stochastic prediction.

2. **Reliability stratification**  
   Candidate-level MC-Dropout variance is used to classify candidates
   as HIGH or LOW reliability.

3. **Interpretable spatial assessment**  
   HIGH-reliability candidates can be evaluated using an
   expert-anchored Spatial Concern Score (SCS), combining local
   uncertainty with spatial proximity information.

## Method

The inference pipeline performs 30 stochastic forward passes with
Dropout3d active at inference time. For each voxel, the mean prediction
and population variance are computed.

Candidates are generated using:

- mean foreground probability threshold: `0.10`
- 6-connected components
- maximum candidate probability threshold: `0.80`

Candidate reliability is determined from mean predictive variance.
The reliability threshold was selected using development data and
frozen before validation.

LOW-reliability candidates represent **abstention**, not rejection or
a negative diagnosis.

## Key Results

On the internal validation cohort:

- 31 candidates were generated.
- 14 candidates were classified as HIGH reliability.
- All 14 HIGH candidates overlapped reference tumors.
- 4 of 17 LOW candidates overlapped reference tumors.
- HIGH candidates had higher local boundary agreement than LOW
  candidates:
  - HIGH median Dice: `0.82`
  - LOW median Dice: `0.62`
  - exploratory Mann--Whitney `p = 0.008`

MC-Dropout variance achieved an overlap AUC of `0.876`, compared with
`0.846` for maximum tumor probability. The difference was not
statistically significant; therefore, uncertainty is treated as a
complementary reliability signal rather than a superior detection
criterion.

## Spatial Concern Assessment

For HIGH-reliability candidates, the Spatial Concern Score combines
normalized local uncertainty with spatial proximity to the portal
vein.

The SCS is an **expert- and literature-informed research prototype**.
It is not a clinically calibrated risk or safety score.

LOW-reliability candidates do not receive an SCS. The framework
explicitly abstains to avoid assigning a quantitative score when
candidate reliability is insufficient.

## License

The source code is provided for research review and reproducibility.
No license for reuse, modification, or redistribution is granted at
this time.
├── scripts/
├── requirements.txt
└── README.md
