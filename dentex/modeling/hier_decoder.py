"""Mask2Former's transformer decoder with extra classification heads.

Stock Mask2Former attaches a single ``nn.Linear(hidden_dim, num_classes + 1)``
to each query and emits ``pred_logits``.  This subclass adds one more linear
head per auxiliary hierarchy level, so a single query can say at once "this is
a *deep caries*" (primary head), "in quadrant *4*" and "at position *7*" --
i.e. tooth 47.

Only two methods are overridden, and the decoder's attention loop is left
untouched:

``forward_prediction_heads``
    also runs the auxiliary heads and buffers their logits;
``forward``
    drains that buffer and attaches the logits to the output dict, including
    the per-layer ``aux_outputs`` entries used for deep supervision.

Weights of the stock heads keep their original names, so a checkpoint trained
with plain Mask2Former (for example the 32-class tooth model from stage 2, or a
public tooth-segmentation checkpoint) loads into this decoder unchanged; only
the new ``aux_class_embed.*`` parameters start from scratch.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from detectron2.config import configurable

from .hierarchy import LevelSpec, resolve_levels

logger = logging.getLogger(__name__)


# Importing this module registers the decoder, so Mask2Former must be on the
# path by the time it is imported -- see dentex/modeling/__init__.py.
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (  # noqa: E402,E501
    MultiScaleMaskedTransformerDecoder,
)
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import (  # noqa: E402,E501
    TRANSFORMER_DECODER_REGISTRY,
)


@TRANSFORMER_DECODER_REGISTRY.register()
class HierarchicalMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    """Mask2Former decoder with auxiliary per-query classification heads."""

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification: bool = True,
        *,
        aux_levels: Sequence[str] = (),
        **kwargs,
    ):
        super().__init__(in_channels, mask_classification, **kwargs)

        self.aux_levels: List[LevelSpec] = list(resolve_levels(aux_levels))
        hidden_dim = kwargs["hidden_dim"]
        self.aux_class_embed = nn.ModuleDict(
            {
                level.name: nn.Linear(hidden_dim, level.num_classes + 1)
                for level in self.aux_levels
            }
        )
        # Filled by forward_prediction_heads, drained by forward.
        self._aux_buffer: List[Dict[str, torch.Tensor]] = []

        if self.aux_levels:
            logger.info(
                "[%s] auxiliary heads: %s",
                type(self).__name__,
                {level.name: level.num_classes for level in self.aux_levels},
            )

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = super().from_config(cfg, in_channels, mask_classification)
        ret["aux_levels"] = list(cfg.MODEL.DENTEX.AUX_HEADS)
        return ret

    # ------------------------------------------------------------------ #

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        outputs_class, outputs_mask, attn_mask = super().forward_prediction_heads(
            output, mask_features, attn_mask_target_size
        )
        if self.aux_levels:
            # Recompute exactly what the parent fed its own head: the normalised
            # decoder output, transposed to [B, Q, C].
            decoder_output = self.decoder_norm(output).transpose(0, 1)
            self._aux_buffer.append(
                {
                    level.name: self.aux_class_embed[level.name](decoder_output)
                    for level in self.aux_levels
                }
            )
        return outputs_class, outputs_mask, attn_mask

    def forward(self, x, mask_features, mask: Optional[torch.Tensor] = None):
        self._aux_buffer = []
        out = super().forward(x, mask_features, mask)

        if not self.aux_levels:
            return out

        buffer, self._aux_buffer = self._aux_buffer, []
        aux_outputs = out.get("aux_outputs") or []
        expected = len(aux_outputs) + 1
        if len(buffer) != expected:
            raise RuntimeError(
                f"collected {len(buffer)} auxiliary prediction(s) but the decoder "
                f"produced {expected} layer output(s); the upstream Mask2Former "
                "decoder loop may have changed"
            )

        for level in self.aux_levels:
            out[level.logits_key] = buffer[-1][level.name]
        for i, aux in enumerate(aux_outputs):
            for level in self.aux_levels:
                aux[level.logits_key] = buffer[i][level.name]
        return out


__all__ = ["HierarchicalMultiScaleMaskedTransformerDecoder"]
