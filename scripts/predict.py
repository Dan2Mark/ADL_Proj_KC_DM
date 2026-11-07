#!/usr/bin/env python
"""Run the full pipeline on x-rays and write annotated images plus a json report.

Example::

    python scripts/predict.py \
        --tooth-config configs/dentex_tooth_R50.yaml \
        --tooth-weights outputs/stage2_tooth/model_final.pth \
        --disease-config configs/dentex_disease_R50_hier.yaml \
        --disease-weights outputs/stage3_disease/model_final.pth \
        --images dataset/dentex_val/validation_data/*/xrays \
        --out outputs/predictions --limit 12 --contact-sheet

For every x-ray this writes ``<name>_annotated.png`` -- thin teal boxes with
FDI numbers for the teeth, thick coloured boxes with the diagnosis for the
findings -- and appends a record to ``report.json``::

    {"image": "...", "n_teeth": 28, "n_problems": 3,
     "problems": [{"problem": "Deep Caries", "tooth_fdi": 47, "score": 0.81, ...}]}

Either model may be omitted: with only ``--tooth-*`` you get numbered teeth,
with only ``--disease-*`` you get findings whose tooth comes from the model's
own enumeration heads.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("predict")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tooth-config", default="")
    parser.add_argument("--tooth-weights", default="")
    parser.add_argument("--disease-config", default="")
    parser.add_argument("--disease-weights", default="")
    parser.add_argument(
        "--images", nargs="+", required=True, help="image files and/or directories"
    )
    parser.add_argument("--out", default="outputs/predictions")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--tooth-score", type=float, default=0.5)
    parser.add_argument("--disease-score", type=float, default=0.5)
    parser.add_argument("--link-iou", type=float, default=0.3)
    parser.add_argument("--no-masks", action="store_true", help="boxes only, lighter output")
    parser.add_argument("--no-teeth-layer", action="store_true", help="draw findings only")
    parser.add_argument("--contact-sheet", action="store_true", help="also write a tiled figure")
    parser.add_argument("--device", default="", help='"cpu" or "cuda"; default is the config')
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def collect_images(patterns: List[str], limit: int = 0) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.is_file():
            found.append(path)
        else:  # treat it as a glob
            found.extend(
                sorted(
                    p
                    for p in Path().glob(pattern)
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
    # Jupyter leaves .ipynb_checkpoints copies next to the data; skip them.
    found = [p for p in found if ".ipynb_checkpoints" not in p.parts]
    seen, unique = set(), []
    for path in found:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique[:limit] if limit else unique


def build_cfg(config_path: str, weights: str, device: str, opts: List[str]):
    from dentex.config import setup_cfg

    cfg = setup_cfg(config_path, opts, freeze=False)
    if weights:
        cfg.MODEL.WEIGHTS = weights
    if device:
        cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.tooth_config and not args.disease_config:
        raise SystemExit("give at least one of --tooth-config / --disease-config")

    opts = list(args.opts)
    if opts and opts[0] == "--":
        opts = opts[1:]

    import dentex.modeling  # noqa: F401  registers the architecture
    from dentex.data.register import register_dentex
    from dentex.inference import DentexPipeline, contact_sheet, draw_report

    tooth_cfg = (
        build_cfg(args.tooth_config, args.tooth_weights, args.device, opts)
        if args.tooth_config
        else None
    )
    disease_cfg = (
        build_cfg(args.disease_config, args.disease_weights, args.device, opts)
        if args.disease_config
        else None
    )

    # DefaultPredictor reads MetadataCatalog for DATASETS.TRAIN[0], so the
    # catalog has to know about the splits even at inference time.
    reference = tooth_cfg or disease_cfg
    register_dentex(
        root=reference.DATASETS.DENTEX_ROOT,
        holdout_root=reference.DATASETS.DENTEX_HOLDOUT_ROOT or None,
    )

    pipeline = DentexPipeline.from_configs(
        tooth_cfg=tooth_cfg,
        disease_cfg=disease_cfg,
        tooth_score_threshold=args.tooth_score,
        disease_score_threshold=args.disease_score,
        link_iou_threshold=args.link_iou,
        keep_masks=not args.no_masks,
    )

    images = collect_images(args.images, args.limit)
    if not images:
        raise SystemExit(f"no images matched {args.images}")
    logger.info("analysing %d x-ray(s)", len(images))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, renders = [], []

    for path in images:
        report = pipeline(path)
        records.append(report.to_dict())
        print(f"\n== {path.name}")
        print(report.to_text())

        rendered = draw_report(
            path,
            report,
            draw_teeth=not args.no_teeth_layer,
            show_masks=not args.no_masks,
            title=path.name,
        )
        target = out_dir / f"{path.stem}_annotated.png"
        rendered.save(target)
        renders.append(rendered)

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    logger.info("wrote %s and %d annotated image(s)", report_path, len(renders))

    if args.contact_sheet and renders:
        sheet = contact_sheet(renders[:8], columns=2, thumb_width=950)
        sheet_path = out_dir / "contact_sheet.png"
        sheet.save(sheet_path)
        logger.info("wrote %s", sheet_path)

    total = sum(r["n_problems"] for r in records)
    logger.info("%d finding(s) across %d x-ray(s)", total, len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
