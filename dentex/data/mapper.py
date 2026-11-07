"""Dataset mapper for DENTEX, built on Mask2Former's instance mapper.

Two things the stock mapper cannot do are needed here.

**Carrying the hierarchy.**  ``annotations_to_instances`` keeps only
``gt_classes`` and ``gt_masks``, but the hierarchical heads need the quadrant,
the position and the diagnosis of every instance.  They are attached to the
``Instances`` object as ``gt_quadrants`` / ``gt_positions`` / ``gt_teeth`` /
``gt_diseases``, which survive batching and indexing.

**Mirror-aware flipping.**  A horizontal flip of a panoramic x-ray is
anatomically meaningful: it swaps the left and right sides of the jaw, so
quadrant 1 becomes quadrant 2 and quadrant 4 becomes quadrant 3.  Flipping
without relabelling teaches the model that tooth 18 can appear on the
lower-left, which is exactly the error we are trying to avoid.  Position inside
the quadrant and the diagnosis are unaffected.  :data:`QUADRANT_MIRROR` encodes
the swap and :class:`DentexDatasetMapper` applies it whenever the sampled
transform list contains an ``HFlipTransform``.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Sequence

import numpy as np
import torch

from .. import fdi

logger = logging.getLogger(__name__)

#: 0-based quadrant index under a left-right mirror: Q1<->Q2 and Q3<->Q4.
QUADRANT_MIRROR = (1, 0, 3, 2)

#: Per-instance hierarchy fields carried through augmentation onto ``Instances``.
HIERARCHY_FIELDS = {
    "quadrant_id": "gt_quadrants",
    "position_id": "gt_positions",
    "tooth_id": "gt_teeth",
    "disease_id": "gt_diseases",
}


def mirror_annotation(ann: dict, task: str) -> dict:
    """Relabel one annotation for a left-right mirrored image (in place)."""
    quadrant_id = ann.get("quadrant_id", -1)
    if quadrant_id >= 0:
        quadrant_id = QUADRANT_MIRROR[quadrant_id]
        ann["quadrant_id"] = quadrant_id
        position_id = ann.get("position_id", -1)
        if position_id >= 0:
            ann["tooth_id"] = fdi.tooth_class_id(quadrant_id + 1, position_id + 1)

    primary = {
        "quadrant": ann.get("quadrant_id", -1),
        "tooth": ann.get("tooth_id", -1),
        "disease": ann.get("disease_id", -1),
    }[task]
    if primary >= 0:
        ann["category_id"] = int(primary)
    return ann


def build_augmentations(cfg, is_train: bool) -> list:
    """Augmentations tuned for panoramic radiographs.

    Rotation and vertical flips are deliberately absent: a panoramic x-ray has a
    fixed orientation, and the model is meant to use that prior (upper jaw on
    top) to decide the quadrant.  What varies between the three source clinics
    is exposure and scale, so that is what gets jittered.
    """
    from detectron2.data import transforms as T

    if not is_train:
        return [
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                sample_style="choice",
            )
        ]

    augs: list = [
        T.ResizeShortestEdge(
            short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
            max_size=cfg.INPUT.MAX_SIZE_TRAIN,
            sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    xray = cfg.INPUT.XRAY
    if xray.BRIGHTNESS > 0:
        augs.append(T.RandomBrightness(1.0 - xray.BRIGHTNESS, 1.0 + xray.BRIGHTNESS))
    if xray.CONTRAST > 0:
        augs.append(T.RandomContrast(1.0 - xray.CONTRAST, 1.0 + xray.CONTRAST))
    if xray.RANDOM_SCALE_RANGE[0] < xray.RANDOM_SCALE_RANGE[1]:
        augs.append(
            T.RandomApply(
                T.ResizeScale(
                    min_scale=xray.RANDOM_SCALE_RANGE[0],
                    max_scale=xray.RANDOM_SCALE_RANGE[1],
                    target_height=cfg.INPUT.MIN_SIZE_TRAIN[-1],
                    target_width=cfg.INPUT.MAX_SIZE_TRAIN,
                ),
                prob=xray.RANDOM_SCALE_PROB,
            )
        )
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    if xray.HFLIP_PROB > 0:
        augs.append(T.RandomFlip(prob=xray.HFLIP_PROB, horizontal=True, vertical=False))
    return augs


class DentexDatasetMapper:
    """Map a DENTEX dataset dict to the tensors Mask2Former trains on.

    Produces ``{"image": CHW uint8 tensor, "instances": Instances}`` where
    ``Instances`` has ``gt_classes``, ``gt_masks`` (a plain ``[N, H, W]`` bool
    tensor, as Mask2Former's criterion expects) and the four hierarchy fields.
    """

    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: Sequence,
        image_format: str,
        task: str,
        size_divisibility: int = -1,
        mirror_aware_flip: bool = True,
    ):
        self.is_train = is_train
        self.augmentations = list(augmentations)
        self.image_format = image_format
        self.task = task
        self.size_divisibility = size_divisibility
        self.mirror_aware_flip = mirror_aware_flip
        logger.info(
            "[DentexDatasetMapper] task=%s train=%s augmentations=%s",
            task, is_train, self.augmentations,
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        return cls(
            is_train,
            augmentations=build_augmentations(cfg, is_train),
            image_format=cfg.INPUT.FORMAT,
            task=cfg.MODEL.DENTEX.TASK,
            size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
            mirror_aware_flip=cfg.INPUT.XRAY.MIRROR_AWARE_FLIP,
        )

    # ------------------------------------------------------------------ #

    def __call__(self, dataset_dict: dict) -> dict:
        from detectron2.data import detection_utils as utils
        from detectron2.data import transforms as T
        from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.augmentations, aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w

        flipped = self.mirror_aware_flip and any(
            isinstance(t, T.HFlipTransform) for t in transforms.transforms
        )

        annotations = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]
        if flipped:
            annotations = [mirror_annotation(a, self.task) for a in annotations]

        masks: List[np.ndarray] = []
        kept: List[dict] = []
        for ann in annotations:
            segmentation = ann.get("segmentation")
            mask = self._to_bitmask(segmentation, image_shape, polygons_to_bitmask)
            if mask is None or not mask.any():
                # Cropping or an aggressive resize can wipe a small lesion out
                # entirely; an all-zero mask makes the dice loss degenerate.
                continue
            masks.append(mask)
            kept.append(ann)

        image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask_tensors = [torch.from_numpy(np.ascontiguousarray(m)) for m in masks]

        if self.size_divisibility > 0:
            image_tensor, mask_tensors = self._pad(image_tensor, mask_tensors)

        padded_shape = (image_tensor.shape[-2], image_tensor.shape[-1])
        dataset_dict["image"] = image_tensor

        instances = Instances(padded_shape)
        instances.gt_classes = torch.tensor(
            [int(a["category_id"]) for a in kept], dtype=torch.int64
        )
        if mask_tensors:
            instances.gt_masks = BitMasks(torch.stack(mask_tensors)).tensor
        else:
            instances.gt_masks = torch.zeros((0, *padded_shape), dtype=torch.bool)
        for key, field in HIERARCHY_FIELDS.items():
            instances.set(
                field,
                torch.tensor([int(a.get(key, -1)) for a in kept], dtype=torch.int64),
            )

        dataset_dict["instances"] = instances
        return dataset_dict

    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_bitmask(segmentation, image_shape, polygons_to_bitmask) -> Optional[np.ndarray]:
        if segmentation is None:
            return None
        if isinstance(segmentation, list):
            parts = [np.asarray(p, dtype=np.float64) for p in segmentation if len(p) >= 6]
            if not parts:
                return None
            return polygons_to_bitmask(parts, *image_shape)
        if isinstance(segmentation, dict):
            import pycocotools.mask as mask_util

            return mask_util.decode(segmentation).astype(bool)
        if isinstance(segmentation, np.ndarray):
            if segmentation.ndim != 2:
                raise ValueError(f"expected a 2D mask, got shape {segmentation.shape}")
            return segmentation.astype(bool)
        raise ValueError(f"cannot turn segmentation of type {type(segmentation)} into a mask")

    def _pad(self, image: torch.Tensor, masks: List[torch.Tensor]):
        import torch.nn.functional as F

        height, width = image.shape[-2], image.shape[-1]
        pad_w = max(self.size_divisibility - width, 0)
        pad_h = max(self.size_divisibility - height, 0)
        if pad_w == 0 and pad_h == 0:
            return image, masks
        padding = [0, pad_w, 0, pad_h]
        image = F.pad(image, padding, value=128).contiguous()
        masks = [F.pad(m, padding, value=0).contiguous() for m in masks]
        return image, masks


__all__ = [
    "HIERARCHY_FIELDS",
    "QUADRANT_MIRROR",
    "DentexDatasetMapper",
    "build_augmentations",
    "mirror_annotation",
]
