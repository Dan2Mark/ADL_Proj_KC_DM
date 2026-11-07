"""The hierarchy levels a DENTEX model can predict, in one table.

Mask2Former has exactly one classification head per query.  DENTEX labels sit
on three levels at once, so the model gets a *primary* head -- the one the
Hungarian matcher and the mask loss are driven by -- plus any number of
*auxiliary* heads that reuse the same matching.

Which level is primary depends on the model:

* the **tooth** detector makes ``tooth`` primary (32 FDI classes), because every
  tooth in the enumeration subset is an object to be found;
* the **pathology** model makes ``disease`` primary (4 classes), because only
  abnormal teeth are annotated there, and adds ``quadrant`` + ``position`` as
  auxiliary heads so each finding also says *which* tooth it sits on.

Keeping the four levels in one table means the decoder, the criterion, the
mapper and the evaluator all agree on names, sizes and tensor fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from .. import fdi


@dataclass(frozen=True)
class LevelSpec:
    """One classification level."""

    name: str
    num_classes: int
    gt_field: str
    """Field name on the ``Instances`` object produced by the dataset mapper."""
    target_key: str
    """Key under which the criterion finds this level's labels in a target dict."""
    class_names: Tuple[str, ...]

    @property
    def logits_key(self) -> str:
        """Key this level's logits use inside the decoder's output dict."""
        return f"pred_logits_{self.name}"

    @property
    def loss_key(self) -> str:
        return f"loss_{self.name}"

    @property
    def pred_field(self) -> str:
        """Field name on the predicted ``Instances``."""
        return f"pred_{self.name}"

    @property
    def score_field(self) -> str:
        return f"pred_{self.name}_score"


LEVELS: Dict[str, LevelSpec] = {
    spec.name: spec
    for spec in (
        LevelSpec(
            "quadrant",
            fdi.NUM_QUADRANTS,
            "gt_quadrants",
            "quadrants",
            fdi.QUADRANT_NAMES,
        ),
        LevelSpec(
            "position",
            fdi.NUM_POSITIONS,
            "gt_positions",
            "positions",
            fdi.POSITION_NAMES,
        ),
        LevelSpec(
            "tooth",
            fdi.NUM_TEETH,
            "gt_teeth",
            "teeth",
            fdi.TOOTH_CLASS_NAMES,
        ),
        LevelSpec(
            "disease",
            fdi.NUM_DISEASES,
            "gt_diseases",
            "diseases",
            fdi.DISEASE_CLASS_NAMES,
        ),
    )
}

#: The level a task's primary head predicts.
TASK_PRIMARY_LEVEL = {
    "quadrant": "quadrant",
    "tooth": "tooth",
    "disease": "disease",
}


def get_level(name: str) -> LevelSpec:
    try:
        return LEVELS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown hierarchy level {name!r}; choose from {sorted(LEVELS)}"
        ) from exc


def resolve_levels(names: Sequence[str]) -> Tuple[LevelSpec, ...]:
    """Validate and de-duplicate a list of level names, preserving order."""
    seen: list[str] = []
    for name in names:
        get_level(name)
        if name not in seen:
            seen.append(name)
    return tuple(LEVELS[name] for name in seen)


__all__ = ["LEVELS", "TASK_PRIMARY_LEVEL", "LevelSpec", "get_level", "resolve_levels"]
