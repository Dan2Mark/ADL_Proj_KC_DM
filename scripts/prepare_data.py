#!/usr/bin/env python
"""Download DENTEX and print the split statistics.

Examples::

    # only what the supervised pipeline needs, streamed from the Hub
    python scripts/prepare_data.py --root dataset/dentex --holdout-root dataset/dentex_val

    # add the 1571 unlabelled x-rays for self-supervised pre-training
    python scripts/prepare_data.py --root dataset/dentex --subsets quadrant enumeration disease unlabelled

    # inspect what is already on disk without downloading anything
    python scripts/prepare_data.py --root dataset/dentex --stats-only

Streaming mode reads the remote zip's central directory and pulls out only the
members it needs, so the 1571-image unlabelled split is never downloaded unless
you ask for it -- about 5 GB instead of 10.9 GB.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dentex.data import convert, download  # noqa: E402
from dentex.data.register import TASK_TO_SUBSET  # noqa: E402

logger = logging.getLogger("prepare_data")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="dataset/dentex", help="where training_data/ goes")
    parser.add_argument(
        "--holdout-root",
        default="dataset/dentex_val",
        help="where the official 50-image validation release goes ('' to skip)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(download.DEFAULT_SUBSETS),
        choices=sorted(download.SUBSET_DIRS),
    )
    parser.add_argument("--mode", default="stream", choices=["stream", "full"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--salt", default="dentex")
    parser.add_argument("--stats-only", action="store_true", help="do not download")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--export-coco",
        default="",
        help="directory to also write plain single-label COCO json files into",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    root = Path(args.root)
    if not args.stats_only:
        download.fetch_training_data(
            root, subsets=args.subsets, mode=args.mode, overwrite=args.overwrite
        )
        if args.holdout_root:
            download.fetch_validation_data(args.holdout_root, overwrite=args.overwrite)

    summary = {}
    for task, subset in TASK_TO_SUBSET.items():
        if subset not in args.subsets:
            continue
        try:
            annotation_json, images_dir = download.locate(root, subset)
        except FileNotFoundError as exc:
            logger.warning("%s not available: %s", task, exc)
            continue
        for split in ("train", "val"):
            records = convert.load_dentex(
                annotation_json,
                images_dir,
                task=task,
                split=split,
                val_fraction=args.val_fraction,
                salt=args.salt,
                require_images=False,
            )
            summary[f"{task}/{split}"] = convert.dataset_statistics(records)
            if args.export_coco:
                convert.to_coco_json(
                    records,
                    task,
                    Path(args.export_coco) / f"dentex_{task}_{split}.json",
                )

    if args.holdout_root:
        try:
            gt_json, images_dir = download.find_validation(args.holdout_root)
        except FileNotFoundError as exc:
            logger.warning("hold-out not available: %s", exc)
        else:
            records = convert.load_dentex(gt_json, images_dir, task="disease")
            summary["disease/holdout"] = convert.dataset_statistics(records)
            if args.export_coco:
                convert.to_coco_json(
                    records, "disease", Path(args.export_coco) / "dentex_disease_holdout.json"
                )

    print()
    print("=" * 78)
    for name, stats in summary.items():
        print(
            f"{name:26s} {stats['images']:5d} images  {stats['instances']:6d} instances  "
            f"{stats['instances_per_image']:6.2f}/image  "
            f"{stats['images_without_annotations']:3d} empty"
        )
    print("=" * 78)

    disease = summary.get("disease/train")
    if disease and disease["diseases"]:
        counts = disease["diseases"]
        print("\ndiagnosis counts in disease/train:")
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:20s} {count:5d}")
        print(
            "\nCLASS_FREQUENCIES.disease for the config "
            "(order: Impacted, Caries, Periapical Lesion, Deep Caries):"
        )
        from dentex import fdi

        print("  " + json.dumps([counts.get(n, 0) for n in fdi.DISEASE_CLASS_NAMES]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
