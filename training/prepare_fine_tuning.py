"""
Align a target nnU-Net plans.json with a pretrained model's plans.json so
that `nnUNetv2_train -pretrained_weights` doesn't fail on an architecture /
patch_size / axis mismatch between the pretrained checkpoint and the
fine-tuning dataset's own auto-generated plan.

By default this transplants "architecture" and "patch_size" - the two keys
that caused the original LiTS -> HCC-TACE axis mismatch. Pass --keys to
align a different/larger set if a future dataset needs more than topology +
patch size aligned, e.g.:

    python prepare_fine_tuning.py source.json target.json \
        --keys architecture patch_size transpose_forward transpose_backward

A backup of the target file is written to <target>.bak before overwriting.

Usage:
    python prepare_fine_tuning.py /path/to/lits/nnUNetPlans.json /path/to/hcc/nnUNetPlans.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_KEYS = ["architecture", "patch_size"]


def align_plans(source_path: Path, target_path: Path, configuration: str, keys: list[str]) -> None:
    if not source_path.exists():
        sys.exit(f"ERROR: source plans not found: {source_path}")
    if not target_path.exists():
        sys.exit(f"ERROR: target plans not found: {target_path}")

    with open(source_path) as f:
        source = json.load(f)
    with open(target_path) as f:
        target = json.load(f)

    for plans, label in [(source, "source"), (target, "target")]:
        if configuration not in plans.get("configurations", {}):
            sys.exit(f"ERROR: configuration '{configuration}' not found in {label} plans")

    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    shutil.copy(target_path, backup_path)
    print(f"Backed up original target plans to {backup_path}")

    src_cfg = source["configurations"][configuration]
    tgt_cfg = target["configurations"][configuration]

    print(f"\nAligning configuration '{configuration}':")
    for key in keys:
        if key not in src_cfg:
            print(f"  [!] skipping '{key}': not present in source configuration")
            continue
        old_value = tgt_cfg.get(key)
        tgt_cfg[key] = src_cfg[key]
        print(f"  {key}: {old_value} -> {src_cfg[key]}")

    with open(target_path, "w") as f:
        json.dump(target, f, indent=4)

    print(f"\nDone. {target_path} now matches {source_path} on: {', '.join(keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source_plans", type=Path, help="plans.json of the pretrained model (e.g. LiTS)")
    parser.add_argument("target_plans", type=Path, help="plans.json of the fine-tuning dataset (e.g. HCC-TACE)")
    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnU-Net configuration to align (default: 3d_fullres)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help=f"plan keys to transplant from source to target (default: {DEFAULT_KEYS})",
    )
    args = parser.parse_args()

    align_plans(args.source_plans, args.target_plans, args.configuration, args.keys)


if __name__ == "__main__":
    main()
