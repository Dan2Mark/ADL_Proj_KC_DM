"""Meta-architecture: Mask2Former that predicts boxes and a label hierarchy.

Three changes on top of ``MaskFormer``.

**Boxes.**  Stock Mask2Former returns ``pred_boxes = Boxes(zeros(N, 4))`` -- the
line that derives boxes from masks is commented out upstream because COCO
instance segmentation is scored on masks.  This project is specified in boxes,
so the box is computed from the thresholded mask here.

**Hierarchy at inference.**  The auxiliary heads added by
:class:`~dentex.modeling.hier_decoder.HierarchicalMultiScaleMaskedTransformerDecoder`
are decoded alongside the primary head and attached to the returned
``Instances`` as ``pred_quadrant`` / ``pred_position`` / ... together with their
confidences.

**Hierarchy at training.**  ``prepare_targets`` forwards the per-instance level
labels produced by :class:`~dentex.data.mapper.DentexDatasetMapper`, and
``from_config`` swaps in :class:`~dentex.modeling.hier_criterion.HierarchicalSetCriterion`.

``forward`` is a near-copy of the upstream one; it has to be, because the
auxiliary logits need to travel from the decoder output into
``instance_inference``, and upstream's signature has no room for them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.matcher import HungarianMatcher

from .hier_criterion import (
    HierarchicalSetCriterion,
    build_hierarchical_weight_dict,
    inverse_frequency_weights,
)
from .hierarchy import TASK_PRIMARY_LEVEL, LevelSpec, resolve_levels

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class HierarchicalMaskFormer(MaskFormer):
    """Mask2Former with box output and auxiliary hierarchy heads."""

    @configurable
    def __init__(
        self,
        *,
        aux_levels=(),
        predict_boxes: bool = True,
        mask_threshold: float = 0.5,
        keep_masks: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aux_levels: List[LevelSpec] = list(resolve_levels(aux_levels))
        self.predict_boxes = predict_boxes
        self.mask_threshold = mask_threshold
        self.keep_masks = keep_masks

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        aux_levels = list(cfg.MODEL.DENTEX.AUX_HEADS)
        aux_weights = dict(zip(aux_levels, cfg.MODEL.DENTEX.AUX_LOSS_WEIGHTS)) if (
            cfg.MODEL.DENTEX.AUX_LOSS_WEIGHTS
        ) else {}
        if aux_levels and len(cfg.MODEL.DENTEX.AUX_LOSS_WEIGHTS) not in (0, len(aux_levels)):
            raise ValueError(
                "MODEL.DENTEX.AUX_LOSS_WEIGHTS must be empty or have one entry per "
                f"MODEL.DENTEX.AUX_HEADS ({len(aux_levels)}), got "
                f"{len(cfg.MODEL.DENTEX.AUX_LOSS_WEIGHTS)}"
            )

        weight_dict = build_hierarchical_weight_dict(
            {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight},
            aux_levels,
            aux_weights,
            cfg.MODEL.MASK_FORMER.DEC_LAYERS,
            deep_supervision,
        )

        losses = ["labels", "masks"]
        if aux_levels:
            losses.append("hierarchy")

        class_weights: Dict[str, torch.Tensor] = {}
        for name, counts in cfg.MODEL.DENTEX.CLASS_FREQUENCIES.items():
            if counts:
                class_weights[name] = inverse_frequency_weights(
                    counts, power=cfg.MODEL.DENTEX.CLASS_WEIGHT_POWER
                )

        criterion = HierarchicalSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            aux_levels=aux_levels,
            aux_eos_coef=(
                None
                if cfg.MODEL.DENTEX.AUX_NO_OBJECT_WEIGHT < 0
                else cfg.MODEL.DENTEX.AUX_NO_OBJECT_WEIGHT
            ),
            class_weights=class_weights,
            primary_level=TASK_PRIMARY_LEVEL[cfg.MODEL.DENTEX.TASK],
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # dentex extras
            "aux_levels": aux_levels,
            "predict_boxes": cfg.MODEL.DENTEX.PREDICT_BOXES,
            "mask_threshold": cfg.MODEL.DENTEX.MASK_THRESHOLD,
            "keep_masks": cfg.MODEL.DENTEX.KEEP_MASKS,
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def prepare_targets(self, targets, images):
        """Upstream targets plus one label tensor per auxiliary level."""
        new_targets = super().prepare_targets(targets, images)
        for target, instances in zip(new_targets, targets):
            for level in self.aux_levels:
                if instances.has(level.gt_field):
                    target[level.target_key] = instances.get(level.gt_field)
                else:
                    # The subset does not label this level -- mark every
                    # instance as "ignore" so the head contributes no gradient.
                    target[level.target_key] = torch.full(
                        (len(instances),), -1, dtype=torch.int64, device=self.device
                    )
        return new_targets

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            losses = self.criterion(outputs, targets)
            for key in list(losses.keys()):
                if key in self.criterion.weight_dict:
                    losses[key] *= self.criterion.weight_dict[key]
                else:
                    losses.pop(key)
            return losses

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = F.interpolate(
            outputs["pred_masks"],
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        aux_logits = {
            level.name: outputs[level.logits_key]
            for level in self.aux_levels
            if level.logits_key in outputs
        }
        del outputs

        processed_results: List[dict] = []
        for index, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(
            zip(mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes)
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            per_image_aux = {
                name: logits[index].to(mask_pred_result) for name, logits in aux_logits.items()
            }

            if self.semantic_on:
                result = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    result = retry_if_cuda_oom(sem_seg_postprocess)(
                        result, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = result

            if self.panoptic_on:
                processed_results[-1]["panoptic_seg"] = retry_if_cuda_oom(
                    self.panoptic_inference
                )(mask_cls_result, mask_pred_result)

            if self.instance_on:
                processed_results[-1]["instances"] = retry_if_cuda_oom(
                    self.hierarchical_instance_inference
                )(mask_cls_result, mask_pred_result, per_image_aux)

        return processed_results

    def hierarchical_instance_inference(
        self,
        mask_cls: torch.Tensor,
        mask_pred: torch.Tensor,
        aux_logits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Instances:
        """Decode queries into instances with boxes, masks and hierarchy labels.

        Args:
            mask_cls: ``[Q, C + 1]`` primary logits.
            mask_pred: ``[Q, H, W]`` mask logits at output resolution.
            aux_logits: ``{level_name: [Q, Cl + 1]}`` auxiliary logits.

        Returns:
            ``Instances`` with ``pred_boxes``, ``scores``, ``pred_classes``,
            optionally ``pred_masks``, and ``pred_<level>`` /
            ``pred_<level>_score`` for every auxiliary level.
        """
        image_size = mask_pred.shape[-2:]
        num_classes = self.sem_seg_head.num_classes

        # [Q, C] -> flat (query, class) scores, keep the top-k pairs.
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        topk = min(self.test_topk_per_image, scores.numel())
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(topk, sorted=False)
        labels_per_image = labels[topk_indices]
        query_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")

        mask_pred = mask_pred[query_indices]
        binary_masks = (mask_pred > self.mask_threshold).float()

        result = Instances(image_size)
        # Average mask confidence inside the predicted mask, as upstream does.
        mask_scores = (mask_pred.sigmoid().flatten(1) * binary_masks.flatten(1)).sum(1) / (
            binary_masks.flatten(1).sum(1) + 1e-6
        )
        result.scores = scores_per_image * mask_scores
        result.pred_classes = labels_per_image

        if self.predict_boxes:
            result.pred_boxes = BitMasks(binary_masks > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(binary_masks.size(0), 4, device=self.device))

        if self.keep_masks:
            result.pred_masks = binary_masks

        for level in self.aux_levels:
            logits = (aux_logits or {}).get(level.name)
            if logits is None:
                continue
            probability = F.softmax(logits, dim=-1)[:, :-1]
            level_scores, level_labels = probability.max(dim=-1)
            result.set(level.pred_field, level_labels[query_indices])
            result.set(level.score_field, level_scores[query_indices])

        return result


__all__ = ["HierarchicalMaskFormer"]
