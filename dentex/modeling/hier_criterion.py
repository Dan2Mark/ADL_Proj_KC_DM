"""Set criterion with extra cross-entropy terms for the auxiliary levels.

The Hungarian matching is unchanged: queries are still matched to ground-truth
instances using the primary class logits plus the mask cost.  Each auxiliary
level then reuses those indices and adds a plain cross-entropy term, so the
quadrant and position heads are trained on exactly the queries that the matcher
decided are real objects.

Two details matter in practice.

**Unknown levels are ignored, not learned as a class.**  The enumeration subset
has no diagnosis and the quadrant subset has no position; those labels arrive
as ``-1`` and are excluded with ``ignore_index`` rather than being folded into
the no-object class, which would teach the head that a real tooth is "nothing".

**Class imbalance is handled per level.**  DENTEX has 2189 caries against 158
periapical lesions, so :func:`inverse_frequency_weights` can turn a label count
into a per-class weight vector that is passed to the relevant head.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from mask2former.modeling.criterion import SetCriterion

from .hierarchy import LevelSpec, resolve_levels

logger = logging.getLogger(__name__)

IGNORE_INDEX = -1


def inverse_frequency_weights(
    counts: Sequence[float],
    power: float = 0.5,
    normalise: bool = True,
) -> torch.Tensor:
    """Per-class weights that damp a long-tailed label distribution.

    ``power=0`` gives uniform weights, ``power=1`` full inverse frequency;
    ``0.5`` (the default) is the usual compromise -- on the DENTEX diagnosis
    counts it lifts periapical lesions by about 3.7x relative to caries instead
    of the 14x that full inverse frequency would apply.

    Args:
        counts: number of training instances per class, in contiguous id order.
        power: exponent applied to the inverse frequency.
        normalise: scale the result so the mean weight is 1.

    Returns:
        A float tensor of length ``len(counts)``.
    """
    counts_t = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1.0)
    weights = counts_t.pow(-float(power))
    if normalise:
        weights = weights / weights.mean()
    return weights


class HierarchicalSetCriterion(SetCriterion):
    """``SetCriterion`` plus one cross-entropy term per auxiliary level."""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: Sequence[str],
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        aux_levels: Sequence[str] = (),
        aux_eos_coef: Optional[float] = None,
        class_weights: Optional[Mapping[str, torch.Tensor]] = None,
        primary_level: Optional[str] = None,
    ):
        super().__init__(
            num_classes,
            matcher,
            weight_dict,
            eos_coef,
            list(losses),
            num_points,
            oversample_ratio,
            importance_sample_ratio,
        )
        self.aux_levels = list(resolve_levels(aux_levels))
        aux_eos_coef = eos_coef if aux_eos_coef is None else aux_eos_coef

        class_weights = dict(class_weights or {})

        # The primary head carries the worst imbalance on the disease subset
        # (2189 caries vs 158 periapical lesions), so it gets weights too.
        primary_weights = class_weights.get(primary_level) if primary_level else None
        if primary_weights is not None:
            primary_weights = torch.as_tensor(primary_weights, dtype=torch.float32)
            if primary_weights.numel() != num_classes:
                raise ValueError(
                    f"class_weights[{primary_level!r}] must have {num_classes} "
                    f"entries to weight the primary head, got {primary_weights.numel()}"
                )
            empty_weight = self.empty_weight.clone()
            empty_weight[:num_classes] = primary_weights
            self.empty_weight.copy_(empty_weight)
            logger.info(
                "[HierarchicalSetCriterion] primary head %r re-weighted: %s",
                primary_level,
                [round(float(w), 3) for w in primary_weights],
            )
        for level in self.aux_levels:
            weights = torch.ones(level.num_classes + 1)
            override = class_weights.get(level.name)
            if override is not None:
                override = torch.as_tensor(override, dtype=torch.float32)
                if override.numel() != level.num_classes:
                    raise ValueError(
                        f"class_weights[{level.name!r}] must have "
                        f"{level.num_classes} entries, got {override.numel()}"
                    )
                weights[: level.num_classes] = override
            weights[-1] = aux_eos_coef
            self.register_buffer(f"empty_weight_{level.name}", weights)

        if self.aux_levels:
            logger.info(
                "[HierarchicalSetCriterion] auxiliary levels: %s",
                [level.name for level in self.aux_levels],
            )

    # ------------------------------------------------------------------ #

    def _empty_weight(self, level: LevelSpec) -> torch.Tensor:
        return getattr(self, f"empty_weight_{level.name}")

    def loss_hierarchy(self, outputs, targets, indices, num_masks) -> Dict[str, torch.Tensor]:
        """Cross-entropy on every auxiliary level, reusing the primary matching."""
        losses: Dict[str, torch.Tensor] = {}
        if not self.aux_levels:
            return losses

        idx = self._get_src_permutation_idx(indices)
        for level in self.aux_levels:
            logits = outputs.get(level.logits_key)
            if logits is None:
                raise KeyError(
                    f"the decoder did not emit {level.logits_key!r}; is "
                    f"MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME set to the "
                    f"hierarchical decoder and {level.name!r} listed in "
                    f"MODEL.DENTEX.AUX_HEADS?"
                )
            logits = logits.float()
            target = torch.full(
                logits.shape[:2],
                level.num_classes,  # the no-object slot
                dtype=torch.int64,
                device=logits.device,
            )
            matched = torch.cat(
                [t[level.target_key][j] for t, (_, j) in zip(targets, indices)]
            ) if len(idx[0]) else torch.zeros(0, dtype=torch.int64, device=logits.device)
            if len(idx[0]):
                target[idx] = matched.to(target.dtype)

            losses[level.loss_key] = F.cross_entropy(
                logits.transpose(1, 2),
                target,
                self._empty_weight(level),
                ignore_index=IGNORE_INDEX,
            )
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        if loss == "hierarchy":
            return self.loss_hierarchy(outputs, targets, indices, num_masks)
        return super().get_loss(loss, outputs, targets, indices, num_masks)


def build_hierarchical_weight_dict(
    base: Mapping[str, float],
    aux_levels: Iterable[str],
    aux_weights: Mapping[str, float],
    dec_layers: int,
    deep_supervision: bool,
) -> Dict[str, float]:
    """Extend Mask2Former's ``weight_dict`` with the auxiliary level terms.

    ``MaskFormer.forward`` drops any loss whose key is missing from
    ``weight_dict``, so every term -- including the per-decoder-layer copies
    suffixed ``_0``, ``_1``, ... -- has to be listed here.
    """
    weight_dict = dict(base)
    for name in aux_levels:
        weight_dict[f"loss_{name}"] = float(aux_weights.get(name, 1.0))

    if deep_supervision:
        extra: Dict[str, float] = {}
        for i in range(dec_layers - 1):
            extra.update({f"{key}_{i}": value for key, value in weight_dict.items()})
        weight_dict.update(extra)
    return weight_dict


__all__ = [
    "HierarchicalSetCriterion",
    "IGNORE_INDEX",
    "build_hierarchical_weight_dict",
    "inverse_frequency_weights",
]
