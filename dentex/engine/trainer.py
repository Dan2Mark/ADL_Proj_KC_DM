"""Training loop for the DENTEX models.

A ``DefaultTrainer`` subclass that wires in four project-specific pieces:

* the DENTEX dataset mapper (hierarchy labels, mirror-aware flips);
* Mask2Former's parameter-group optimiser, which the architecture needs --
  a flat AdamW over all parameters trains noticeably worse;
* repeat-factor sampling, so the 158 periapical lesions are seen more often
  than their 4.5% share of the disease subset would give them;
* COCO AP plus the hierarchical metrics as the evaluator.

:func:`load_stage_weights` implements the multi-stage transfer that the DENTEX
baseline uses -- quadrant -> enumeration -> diagnosis -- by copying a previous
stage's weights and dropping only the heads whose shape changed.
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import torch

from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.solver.build import maybe_add_gradient_clipping

from ..data.mapper import DentexDatasetMapper
from .evaluator import build_evaluator

logger = logging.getLogger(__name__)


class DentexTrainer(DefaultTrainer):
    """``DefaultTrainer`` with the DENTEX data, optimiser and metrics."""

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        mapper = DentexDatasetMapper.from_config(cfg, is_train=True)
        threshold = cfg.DATALOADER.REPEAT_THRESHOLD_DENTEX
        if threshold <= 0:
            return build_detection_train_loader(cfg, mapper=mapper)

        from detectron2.data.build import get_detection_dataset_dicts
        from detectron2.data.samplers import RepeatFactorTrainingSampler

        # Build the dataset explicitly and derive the repeat factors from the
        # *same* list the loader will index into.  Letting the loader rebuild
        # it from cfg would re-apply FILTER_EMPTY_ANNOTATIONS and leave the
        # sampler indexing a list of a different length.
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=0,
            proposal_files=None,
        )
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset, threshold
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        logger.info(
            "repeat-factor sampling on (threshold=%.2f) over %d images: "
            "factors in [%.2f, %.2f]",
            threshold, len(dataset),
            float(repeat_factors.min()), float(repeat_factors.max()),
        )
        return build_detection_train_loader(
            dataset=dataset,
            mapper=mapper,
            sampler=sampler,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        mapper = DentexDatasetMapper.from_config(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder: Optional[str] = None):
        if output_folder is None:
            import os

            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_lr_scheduler(cls, cfg: CfgNode, optimizer):
        from detectron2.solver import build_lr_scheduler

        return build_lr_scheduler(cfg, optimizer)

    # ------------------------------------------------------------------ #
    # Mask2Former's optimiser, kept verbatim in spirit
    # ------------------------------------------------------------------ #

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        seen: Set[torch.nn.Parameter] = set()
        for module_name, module in model.named_modules():
            for param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in seen:
                    continue
                seen.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in param_name
                    or "absolute_pos_embed" in param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_full_model_clipping(optimizer_class):
            clip_value = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enabled = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_value > 0.0
            )
            if not enabled:
                return optimizer_class

            class FullModelGradientClippingOptimizer(optimizer_class):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[group["params"] for group in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_value)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_full_model_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_full_model_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"unsupported SOLVER.OPTIMIZER {optimizer_type!r}")

        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


# --------------------------------------------------------------------- #
# Multi-stage transfer
# --------------------------------------------------------------------- #

def inspect_transfer(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> Dict[str, List[str]]:
    """Report what a checkpoint would and would not transfer into ``model``.

    Going from the quadrant stage (4 classes) to the enumeration stage (32), or
    from enumeration to diagnosis (4 classes plus auxiliary heads), changes the
    shape of ``class_embed`` and adds ``aux_class_embed``.  Everything else --
    backbone, pixel decoder, the whole transformer decoder -- transfers, and
    that is where almost all of the useful pre-training sits.

    Call this *before* training to see exactly which tensors will be randomly
    initialised, instead of discovering it from detectron2's log wall.

    Args:
        model: the freshly built model the weights would be loaded into.
        checkpoint_path: a ``.pth`` or ``.pkl`` whose keys follow Mask2Former's
            naming.

    Returns:
        ``{"matched": [...], "shape_mismatch": [...], "missing_in_checkpoint":
        [...], "unused_in_checkpoint": [...]}``.
    """
    source = _read_state_dict(checkpoint_path)
    target = model.state_dict()

    report: Dict[str, List[str]] = {
        "matched": [],
        "shape_mismatch": [],
        "missing_in_checkpoint": [],
        "unused_in_checkpoint": [],
    }
    for key, tensor in target.items():
        if key not in source:
            report["missing_in_checkpoint"].append(key)
        elif tuple(source[key].shape) != tuple(tensor.shape):
            report["shape_mismatch"].append(
                f"{key}: checkpoint {tuple(source[key].shape)} vs model {tuple(tensor.shape)}"
            )
        else:
            report["matched"].append(key)
    report["unused_in_checkpoint"] = [k for k in source if k not in target]

    logger.info(
        "transfer from %s: %d matched, %d shape-mismatched, %d missing, %d unused",
        checkpoint_path,
        len(report["matched"]),
        len(report["shape_mismatch"]),
        len(report["missing_in_checkpoint"]),
        len(report["unused_in_checkpoint"]),
    )
    return report


def _read_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    import pickle
    from pathlib import Path

    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            blob = pickle.load(handle, encoding="latin1")
    else:
        blob = torch.load(path, map_location="cpu")
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    return {k: torch.as_tensor(v) for k, v in state.items()}


def filter_checkpoint(
    checkpoint_path: str,
    output_path: str,
    drop_substrings: Sequence[str] = ("class_embed", "static_query", "query_feat"),
) -> str:
    """Write a copy of a checkpoint with the listed parameters removed.

    Use this when the next stage has a different number of classes and you want
    detectron2 to initialise those heads from scratch without printing a wall of
    shape-mismatch warnings.  The ``static_query``/``query_feat`` defaults are
    there because changing the number of object queries also invalidates them.
    """
    import pickle
    from pathlib import Path

    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            blob = pickle.load(handle, encoding="latin1")
        model_state = blob["model"]
        kept = {k: v for k, v in model_state.items() if not any(s in k for s in drop_substrings)}
        blob["model"] = kept
        with open(output_path, "wb") as handle:
            pickle.dump(blob, handle)
    else:
        blob = torch.load(path, map_location="cpu")
        model_state = blob.get("model", blob)
        kept = {k: v for k, v in model_state.items() if not any(s in k for s in drop_substrings)}
        if "model" in blob:
            blob["model"] = kept
        else:
            blob = kept
        torch.save(blob, output_path)

    removed = len(model_state) - len(kept)
    logger.info("filtered %s -> %s (%d tensors removed)", checkpoint_path, output_path, removed)
    return output_path


__all__ = ["DentexTrainer", "filter_checkpoint", "inspect_transfer"]
