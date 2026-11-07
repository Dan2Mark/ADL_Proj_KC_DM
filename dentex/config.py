"""Config additions for the DENTEX models, and the ``setup_cfg`` entry point.

Everything project-specific lives under two new nodes:

``MODEL.DENTEX``
    which task the model is being trained for, which auxiliary hierarchy heads
    it carries, how their losses are weighted and balanced, and whether
    inference should emit boxes and masks.

``INPUT.XRAY``
    augmentation knobs specific to panoramic radiographs.

Everything else -- backbone, pixel decoder, Mask2Former hyper-parameters -- is
whatever ``add_maskformer2_config`` already defines, so stock Mask2Former yaml
files stay loadable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg

logger = logging.getLogger(__name__)


def add_dentex_config(cfg: CN) -> CN:
    """Add the ``MODEL.DENTEX`` and ``INPUT.XRAY`` nodes to ``cfg`` in place."""
    # ---------------------------------------------------------------- #
    # Task and hierarchy
    # ---------------------------------------------------------------- #
    cfg.MODEL.DENTEX = CN()

    #: "quadrant", "tooth" or "disease".  Decides what the primary head means
    #: and which labels the dataset mapper writes into ``category_id``.
    cfg.MODEL.DENTEX.TASK = "tooth"

    #: Extra per-query classification heads, e.g. ["quadrant", "position"] for
    #: the pathology model so each finding also names its tooth.
    cfg.MODEL.DENTEX.AUX_HEADS = []

    #: Loss weight per auxiliary head; empty means 1.0 for all of them.
    cfg.MODEL.DENTEX.AUX_LOSS_WEIGHTS = []

    #: No-object weight for the auxiliary heads; -1 reuses
    #: MODEL.MASK_FORMER.NO_OBJECT_WEIGHT.
    cfg.MODEL.DENTEX.AUX_NO_OBJECT_WEIGHT = -1.0

    #: Training-set label counts per level, used to damp class imbalance.
    #: Fill from dentex.data.convert.dataset_statistics; an empty list means
    #: "do not re-weight this level".
    cfg.MODEL.DENTEX.CLASS_FREQUENCIES = CN(
        {"quadrant": [], "position": [], "tooth": [], "disease": []}
    )

    #: Exponent on the inverse frequency; 0 = uniform, 1 = full inverse.
    cfg.MODEL.DENTEX.CLASS_WEIGHT_POWER = 0.5

    # ---------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------- #

    #: Derive a bounding box from each predicted mask.  Upstream Mask2Former
    #: returns zero boxes; this project is specified in boxes, so keep it on.
    cfg.MODEL.DENTEX.PREDICT_BOXES = True

    #: Threshold on the mask logits when binarising.
    cfg.MODEL.DENTEX.MASK_THRESHOLD = 0.5

    #: Keep ``pred_masks`` on the returned instances.  Turning this off saves a
    #: lot of memory when only boxes are needed.
    cfg.MODEL.DENTEX.KEEP_MASKS = True

    #: Score threshold the hierarchical evaluator and the inference pipeline
    #: apply before matching predictions to ground truth.
    cfg.MODEL.DENTEX.EVAL_SCORE_THRESHOLD = 0.5

    # ---------------------------------------------------------------- #
    # Sampling
    # ---------------------------------------------------------------- #

    #: Oversample images containing rare classes (detectron2's repeat-factor
    #: sampler).  0 disables it; 0.3 is a reasonable value for the disease
    #: subset, where periapical lesions are 14x rarer than caries.
    cfg.DATALOADER.REPEAT_THRESHOLD_DENTEX = 0.0

    # ---------------------------------------------------------------- #
    # X-ray augmentation
    # ---------------------------------------------------------------- #
    cfg.INPUT.XRAY = CN()

    #: Multiplicative brightness jitter, +/- this fraction.  The three source
    #: clinics used different exposure settings, so this is the main nuisance
    #: variation to simulate.
    cfg.INPUT.XRAY.BRIGHTNESS = 0.2
    cfg.INPUT.XRAY.CONTRAST = 0.2

    #: Horizontal flip probability.  Safe only with MIRROR_AWARE_FLIP on.
    cfg.INPUT.XRAY.HFLIP_PROB = 0.5

    #: Relabel quadrants when the image is mirrored (Q1<->Q2, Q3<->Q4).
    #: Without this, flipping teaches the model that tooth 18 can appear in the
    #: lower-left, which destroys enumeration accuracy.
    cfg.INPUT.XRAY.MIRROR_AWARE_FLIP = True

    #: Optional random rescale on top of ResizeShortestEdge.
    cfg.INPUT.XRAY.RANDOM_SCALE_RANGE = [1.0, 1.0]
    cfg.INPUT.XRAY.RANDOM_SCALE_PROB = 0.0

    # ---------------------------------------------------------------- #
    # Data roots, so a config is self-contained
    # ---------------------------------------------------------------- #
    cfg.DATASETS.DENTEX_ROOT = "dataset/dentex"
    cfg.DATASETS.DENTEX_HOLDOUT_ROOT = "dataset/dentex_val"
    cfg.DATASETS.DENTEX_VAL_FRACTION = 0.1
    cfg.DATASETS.DENTEX_SPLIT_SALT = "dentex"

    return cfg


def setup_cfg(
    config_file: Optional[str | Path] = None,
    opts: Optional[Sequence[str]] = None,
    freeze: bool = True,
) -> CN:
    """Build a config with detectron2 + Mask2Former + DENTEX defaults applied.

    Args:
        config_file: yaml to merge on top of the defaults.
        opts: ``["KEY", "value", ...]`` command-line style overrides.
        freeze: freeze the config before returning it.

    Returns:
        The assembled ``CfgNode``.
    """
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_dentex_config(cfg)

    if config_file is not None:
        cfg.merge_from_file(str(config_file))
    if opts:
        cfg.merge_from_list(list(opts))

    validate_cfg(cfg)
    if freeze:
        cfg.freeze()
    return cfg


def validate_cfg(cfg: CN) -> None:
    """Catch the configuration mistakes that only surface hours into training."""
    from .data.convert import TASK_NUM_CLASSES, TASKS
    from .modeling.hierarchy import TASK_PRIMARY_LEVEL, get_level

    task = cfg.MODEL.DENTEX.TASK
    if task not in TASKS:
        raise ValueError(f"MODEL.DENTEX.TASK must be one of {TASKS}, got {task!r}")

    expected = TASK_NUM_CLASSES[task]
    if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES != expected:
        raise ValueError(
            f"MODEL.SEM_SEG_HEAD.NUM_CLASSES is "
            f"{cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES} but task {task!r} has "
            f"{expected} classes"
        )

    primary = TASK_PRIMARY_LEVEL[task]
    for name in cfg.MODEL.DENTEX.AUX_HEADS:
        level = get_level(name)
        if name == primary:
            raise ValueError(
                f"{name!r} is already the primary head for task {task!r}; "
                "remove it from MODEL.DENTEX.AUX_HEADS"
            )
        counts = cfg.MODEL.DENTEX.CLASS_FREQUENCIES.get(name, [])
        if counts and len(counts) != level.num_classes:
            raise ValueError(
                f"MODEL.DENTEX.CLASS_FREQUENCIES.{name} has {len(counts)} "
                f"entries but level {name!r} has {level.num_classes} classes"
            )

    counts = cfg.MODEL.DENTEX.CLASS_FREQUENCIES.get(primary, [])
    if counts and len(counts) != expected:
        raise ValueError(
            f"MODEL.DENTEX.CLASS_FREQUENCIES.{primary} has {len(counts)} "
            f"entries but the primary head has {expected} classes"
        )

    if cfg.MODEL.DENTEX.AUX_HEADS and not cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME.startswith(
        "Hierarchical"
    ):
        raise ValueError(
            "MODEL.DENTEX.AUX_HEADS is set but "
            "MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME is "
            f"{cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME!r}; use "
            "'HierarchicalMultiScaleMaskedTransformerDecoder'"
        )

    if cfg.INPUT.XRAY.HFLIP_PROB > 0 and not cfg.INPUT.XRAY.MIRROR_AWARE_FLIP:
        logger.warning(
            "horizontal flips are on with MIRROR_AWARE_FLIP off -- quadrant "
            "labels will be wrong on every flipped sample"
        )

    if cfg.MODEL.DENTEX.AUX_NO_OBJECT_WEIGHT < 0:
        # Resolved to the primary head's value by the meta-architecture.
        pass


__all__ = ["add_dentex_config", "setup_cfg", "validate_cfg"]
