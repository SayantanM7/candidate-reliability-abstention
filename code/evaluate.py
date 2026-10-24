"""Validation results of the paper (20 cases).

  Table 1  candidate-level reliability with exact (Clopper-Pearson) 95% intervals
  Local Dice between each lesion and its coinciding candidate, HIGH vs LOW (exact Mann-Whitney)
  Table 2  mean Dice / precision / recall for the raw prediction, Stage 1 and Stage 2
  AUC for tumour overlap: -mean variance vs maximum probability, paired case-level bootstrap

Usage: python code/evaluate.py              # from the NIfTI outputs; also rewrites results/*.csv
       python code/evaluate.py --from-csv   # Table 1, local Dice and AUCs from results/*.csv only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import beta, mannwhitneyu
from sklearn.metrics import roc_auc_score

from candidates import candidate_rows
from common import REPO, case_file, gt_path, load_params, load_split, read


def exact_ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


def table1(v):
    high, ov = v.confidence == "HIGH", v.gt_overlap == 1
    rows = [("HIGH coverage of candidates", high.sum(), len(v)),
            ("Overlap, HIGH candidates", (high & ov).sum(), high.sum()),
            ("Overlap, LOW candidates", (~high & ov).sum(), (~high).sum()),
            ("Overlapping labelled HIGH", (high & ov).sum(), ov.sum()),
            ("Non-overlapping labelled LOW", (~high & ~ov).sum(), (~ov).sum())]
    print("\nTable 1 (validation candidates, exact 95% CI)")
    for name, k, n in rows:
        lo, hi = exact_ci(int(k), int(n))
        print(f"  {name:30s} {k}/{n} ({100 * k / n:.1f}%)  CI {lo:.1f}-{hi:.1f}")


def pair_rows(prm, case, raw_thr, lesion_conn):
    """Every (lesion, candidate) pair in which any candidate voxel lies in the lesion."""
    s1 = np.asarray(read(case_file(prm, case, "stage1"))).astype(np.int32)
    s2 = read(case_file(prm, case, "stage2"))
    raw = read(case_file(prm, case, "mean_prob")) >= raw_thr
    gt = read(gt_path(prm, case)) > 0
    lesions, n = ndi.label(gt, structure=ndi.generate_binary_structure(3, lesion_conn))
    rows = []
    for k in range(1, n + 1):
        L = lesions == k
        ids, cnt = np.unique(s1[L], return_counts=True)
        for b, ov in zip(ids, cnt):
            if b == 0:
                continue
            B = (s1 == b) & raw                                    # candidate's p >= 0.5 voxels
            inter = int((B & L).sum())
            rows.append(dict(case_id=case, lesion_id=k, blob_id=int(b),
                             confidence="HIGH" if int(s2[s1 == b][0]) == 1 else "LOW",
                             lesion_voxels=int(L.sum()), blob_voxels_p05=int(B.sum()),
                             overlap_any_voxels=int(ov), overlap_p05_voxels=inter,
                             dice=2 * inter / (B.sum() + L.sum()), coverage=inter / L.sum(),
                             precision=inter / B.sum() if B.sum() else np.nan))
    return rows


def local_dice(pairs, min_voxels):
    p = pairs[pairs.lesion_voxels >= min_voxels]
    h, l = p[p.confidence == "HIGH"], p[p.confidence == "LOW"]
    pv = mannwhitneyu(h.dice, l.dice, alternative="two-sided", method="exact").pvalue
    print(f"\nLocal Dice ({len(p)} lesion-candidate pairs, lesions >= {min_voxels} voxels)")
    for name, col in (("Dice", "dice"), ("coverage", "coverage"), ("precision", "precision")):
        print(f"  median {name:9s} HIGH {h[col].median():.2f} (n={len(h)})  LOW {l[col].median():.2f} (n={len(l)})")
    print(f"  exact two-sided Mann-Whitney p = {pv:.3f} (exploratory)")


def stage_dice(prm, val, cand, raw_thr):
    rows = []
    for case in val:
        s1 = np.asarray(read(case_file(prm, case, "stage1"))).astype(np.int32)
        raw = read(case_file(prm, case, "mean_prob")) >= raw_thr
        gt = read(gt_path(prm, case)) > 0
        high_ids = cand[(cand.case_id == case) & (cand.confidence == "HIGH")].blob_id.tolist()
        for stage, pred in (("Raw", raw), ("Stage 1", raw & (s1 > 0)), ("Stage 2", raw & np.isin(s1, high_ids))):
            tp, npred = int((pred & gt).sum()), int(pred.sum())
            rows.append(dict(case_id=case, stage=stage, high_case=len(high_ids) > 0,
                             dice=2 * tp / (npred + gt.sum()), recall=tp / gt.sum(),
                             precision=tp / npred if npred else np.nan))   # empty prediction: Dice, recall 0
    df = pd.DataFrame(rows)
    print(f"\nTable 2 ({len(val)} validation cases; abstained cases score 0 in the overall row)")
    for stage, s in df.groupby("stage", sort=False):
        abst = "-" if stage == "Stage 2" else f"{s[~s.high_case].dice.mean():.3f}"
        print(f"  {stage:8s} Dice HIGH cases {s[s.high_case].dice.mean():.3f} (n={s.high_case.sum()}) | "
              f"abstained {abst} | all {s.dice.mean():.3f} +- {s.dice.std():.3f} | "
              f"precision {s.precision.mean():.3f} (n={s.precision.notna().sum()}) | recall {s.recall.mean():.3f}")


def auc_comparison(v, n_boot, seed):
    auc = lambda d: (roc_auc_score(d.gt_overlap, -d.mean_variance), roc_auc_score(d.gt_overlap, d.max_prob))
    a_var, a_max = auc(v)
    rng = np.random.default_rng(seed)
    cases = v.case_id.unique()
    groups = {c: v[v.case_id == c] for c in cases}
    diffs = []
    for _ in range(n_boot):                                      # resample cases, keep their candidates
        b = pd.concat([groups[c] for c in rng.choice(cases, len(cases))], ignore_index=True)
        if b.gt_overlap.nunique() == 2:
            diffs.append(np.subtract(*auc(b)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nAUC for tumour overlap: variance {a_var:.3f}, maximum probability {a_max:.3f}")
    print(f"  paired case-level bootstrap of the difference ({len(diffs)} resamples): "
          f"mean {np.mean(diffs):+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    prm, ev = load_params(), load_params()["evaluation"]
    train, val = load_split()
    out = REPO / prm["paths"]["results_dir"]

    if args.from_csv:
        cand = pd.read_csv(out / "candidate_table_all.csv")
        pairs = pd.read_csv(out / "pair_table.csv")
    else:
        cand = pd.DataFrame([r for split, cases in (("development", train), ("validation", val))
                             for c in cases for r in candidate_rows(prm, c, split)])
        pairs = pd.DataFrame([r for c in val for r in pair_rows(prm, c, ev["raw_threshold"],
                                                                 ev["lesion_connectivity"])])
        out.mkdir(parents=True, exist_ok=True)
        cand.to_csv(out / "candidate_table_all.csv", index=False)
        pairs.to_csv(out / "pair_table.csv", index=False)

    v = cand[cand.split == "validation"]
    print(f"Validation: {len(v)} candidates in {v.case_id.nunique()} of {len(val)} cases")
    table1(v)
    local_dice(pairs, ev["min_lesion_voxels"])
    if not args.from_csv:
        stage_dice(prm, val, cand, ev["raw_threshold"])
    auc_comparison(v, ev["n_bootstrap"], ev["seed"])


if __name__ == "__main__":
    main()
