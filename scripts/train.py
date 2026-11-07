#!/usr/bin/env python
"""Train one stage of the DENTEX pipeline.

Examples::

    # stage 2: tooth detection with FDI enumeration
    python scripts/train.py --config configs/dentex_tooth_R50.yaml \
        --data-root dataset/dentex --holdout-root dataset/dentex_val

    # stage 3: hierarchical pathology, warm-started from stage 2
    python scripts/train.py --config configs/dentex_disease_R50_hier.yaml \
        --data-root dataset/dentex --holdout-root dataset/dentex_val \
        --init-from outputs/stage2_tooth/model_final.pth

    # evaluate a trained stage on the official hold-out
    python scripts/train.py --config configs/dentex_disease_R50_hier.yaml \
        --data-root dataset/dentex --holdout-root dataset/dentex_val \
        --eval-only MODEL.WEIGHTS outputs/stage3_disease/model_final.pth \
        DATASETS.TEST '("dentex_disease_holdout",)'

``--init-from`` is the multi-stage transfer: it copies the checkpoint without
the classification head (whose shape changes between stages) so the backbone,
pixel decoder and transformer decoder all carry over while the new heads start
fresh.  Anything after the named flags is passed straight to detectron2 as
``KEY value`` config overrides.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("train")

#: Parameters whose shape changes between stages and must not transfer.
STAGE_TRANSFER_DROP = ("class_embed", "aux_class_embed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="dataset/dentex")
    parser.add_argument("--holdout-root", default="dataset/dentex_val")
    parser.add_argument(
        "--init-from",
        default="",
        help="checkpoint from a previous stage; its class heads are stripped",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the config, datasets and model, then exit without training",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="KEY value config overrides")
    args = parser.parse_args(argv)
    # detectron2's default_setup logs args.config_file when it is present.
    args.config_file = args.config
    return args


def build(args):
    """Assemble the config and register the datasets it refers to."""
    from dentex.config import setup_cfg
    from dentex.data.register import register_dentex
    import dentex.modeling  # noqa: F401  registers the architecture

    opts = list(args.opts)
    if opts and opts[0] == "--":
        opts = opts[1:]

    cfg = setup_cfg(args.config, opts, freeze=False)
    cfg.DATASETS.DENTEX_ROOT = args.data_root
    cfg.DATASETS.DENTEX_HOLDOUT_ROOT = args.holdout_root

    register_dentex(
        root=cfg.DATASETS.DENTEX_ROOT,
        val_fraction=cfg.DATASETS.DENTEX_VAL_FRACTION,
        salt=cfg.DATASETS.DENTEX_SPLIT_SALT,
        holdout_root=cfg.DATASETS.DENTEX_HOLDOUT_ROOT or None,
    )

    if args.init_from:
        from dentex.engine.trainer import filter_checkpoint

        stripped = Path(cfg.OUTPUT_DIR) / "init_from_previous_stage.pth"
        stripped.parent.mkdir(parents=True, exist_ok=True)
        filter_checkpoint(args.init_from, str(stripped), drop_substrings=STAGE_TRANSFER_DROP)
        cfg.MODEL.WEIGHTS = str(stripped)
        logger.info("warm-starting from %s (class heads dropped)", args.init_from)

    cfg.freeze()
    return cfg


def main_worker(args):
    from detectron2.engine import default_setup, launch  # noqa: F401
    from detectron2.utils import comm

    from dentex.engine.trainer import DentexTrainer, inspect_transfer

    cfg = build(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)

    if args.dry_run:
        from detectron2.modeling import build_model

        model = build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model built: %.1fM trainable parameters", n_params / 1e6)
        if cfg.MODEL.WEIGHTS and Path(cfg.MODEL.WEIGHTS).is_file():
            report = inspect_transfer(model, cfg.MODEL.WEIGHTS)
            for key in report["shape_mismatch"][:10]:
                logger.info("  shape mismatch: %s", key)
        loader = DentexTrainer.build_train_loader(cfg)
        batch = next(iter(loader))
        instances = batch[0]["instances"]
        logger.info(
            "first batch ok: image %s, %d instances, fields %s",
            tuple(batch[0]["image"].shape), len(instances), sorted(instances.get_fields()),
        )
        return None

    if args.eval_only:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.modeling import build_model

        model = build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        results = DentexTrainer.test(cfg, model)
        if comm.is_main_process():
            print(results)
        return results

    trainer = DentexTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from detectron2.engine import launch

    launch(
        main_worker,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args,),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
