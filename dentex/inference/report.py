"""Structured findings: the objects the pipeline returns and how they are reported.

The whole point of the project is an answer of the form *"deep caries on tooth
47"* -- a diagnosis bound to a specific tooth, both with a box.  Everything
downstream (visualiser, json export, markdown summary for the report) consumes
the two dataclasses defined here, so the model-specific code never leaks into
presentation.

Boxes are always ``XYXY`` in absolute pixels of the original x-ray, matching
detectron2's ``Boxes`` convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import fdi

BBox = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #

@dataclass
class ToothFinding:
    """One detected tooth: where it is and which tooth it is."""

    bbox: BBox
    score: float
    tooth_id: int
    """Contiguous id in ``[0, 32)``; see :mod:`dentex.fdi`."""
    mask: Optional[np.ndarray] = None
    """Binary mask at original image resolution, or ``None`` if not kept."""

    @property
    def fdi(self) -> int:
        return fdi.fdi_from_tooth_class_id(self.tooth_id)

    @property
    def quadrant(self) -> int:
        return fdi.split_tooth_class_id(self.tooth_id)[0]

    @property
    def position(self) -> int:
        return fdi.split_tooth_class_id(self.tooth_id)[1]

    @property
    def name(self) -> str:
        return fdi.tooth_long_name(self.fdi)

    def to_dict(self, with_mask: bool = False) -> dict:
        out = {
            "fdi": self.fdi,
            "quadrant": self.quadrant,
            "position": self.position,
            "name": self.name,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox],
            "score": round(float(self.score), 4),
        }
        if with_mask and self.mask is not None:
            out["mask_area_px"] = int(self.mask.sum())
        return out


@dataclass
class ProblemFinding:
    """One detected pathology, with the tooth it was attributed to.

    ``bbox`` is the pathology model's own box.  In DENTEX the annotated object
    *is* the abnormal tooth, so this box outlines the diseased tooth rather than
    an isolated lesion inside it -- see the README.
    """

    bbox: BBox
    score: float
    disease_id: int
    """Contiguous id in ``[0, 4)``; see :data:`dentex.fdi.DISEASE_CLASS_NAMES`."""
    tooth_id: Optional[int] = None
    """Tooth this problem was attributed to, or ``None`` if nothing matched."""
    tooth_source: str = "none"
    """How the tooth was decided: ``"head"``, ``"linked"``, ``"agreed"`` or ``"none"``."""
    link_iou: float = 0.0
    """IoU with the matched tooth box from the tooth detector."""
    head_tooth_id: Optional[int] = None
    """What the pathology model's own enumeration head predicted."""
    mask: Optional[np.ndarray] = None

    @property
    def disease(self) -> str:
        return fdi.DISEASE_CLASS_NAMES[self.disease_id]

    @property
    def fdi(self) -> Optional[int]:
        if self.tooth_id is None:
            return None
        return fdi.fdi_from_tooth_class_id(self.tooth_id)

    def summary(self) -> str:
        """``'Deep Caries on tooth 47 (lower-right second molar), score 0.81'``."""
        where = fdi.tooth_long_name(self.fdi) if self.fdi is not None else "an unmatched tooth"
        return f"{self.disease} on {where}, score {self.score:.2f}"

    def to_dict(self, with_mask: bool = False) -> dict:
        out = {
            "problem": self.disease,
            "tooth_fdi": self.fdi,
            "tooth_name": fdi.tooth_long_name(self.fdi) if self.fdi is not None else None,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox],
            "score": round(float(self.score), 4),
            "tooth_source": self.tooth_source,
            "link_iou": round(float(self.link_iou), 3),
            "head_tooth_fdi": (
                fdi.fdi_from_tooth_class_id(self.head_tooth_id)
                if self.head_tooth_id is not None
                else None
            ),
        }
        if with_mask and self.mask is not None:
            out["mask_area_px"] = int(self.mask.sum())
        return out


@dataclass
class StudyReport:
    """Everything the pipeline found on one x-ray."""

    image_path: str
    width: int
    height: int
    teeth: List[ToothFinding] = field(default_factory=list)
    problems: List[ProblemFinding] = field(default_factory=list)

    # -- views ------------------------------------------------------------- #

    def problems_by_tooth(self) -> Dict[Optional[int], List[ProblemFinding]]:
        """``{fdi_number: [problems]}``, with ``None`` collecting unmatched ones."""
        grouped: Dict[Optional[int], List[ProblemFinding]] = {}
        for problem in self.problems:
            grouped.setdefault(problem.fdi, []).append(problem)
        for items in grouped.values():
            items.sort(key=lambda p: -p.score)
        return dict(sorted(grouped.items(), key=lambda kv: (kv[0] is None, kv[0])))

    def to_dict(self, with_masks: bool = False) -> dict:
        return {
            "image": self.image_path,
            "width": self.width,
            "height": self.height,
            "n_teeth": len(self.teeth),
            "n_problems": len(self.problems),
            "teeth": [t.to_dict(with_masks) for t in self.teeth],
            "problems": [p.to_dict(with_masks) for p in self.problems],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """A short human-readable verdict, suitable for pasting into the report."""
        lines = [f"**{self.image_path}** -- {len(self.teeth)} teeth detected."]
        if not self.problems:
            lines.append("")
            lines.append("No pathology detected above the score threshold.")
            return "\n".join(lines)

        lines += ["", "| Tooth | Problem | Score | Tooth from | IoU |", "|---|---|---|---|---|"]
        for tooth_fdi, problems in self.problems_by_tooth().items():
            label = str(tooth_fdi) if tooth_fdi is not None else "unmatched"
            for problem in problems:
                lines.append(
                    f"| {label} | {problem.disease} | {problem.score:.2f} "
                    f"| {problem.tooth_source} | {problem.link_iou:.2f} |"
                )
        return "\n".join(lines)

    def to_text(self) -> str:
        """Plain-text verdict, one line per finding."""
        if not self.problems:
            return f"{len(self.teeth)} teeth detected, no pathology found."
        head = f"{len(self.teeth)} teeth detected, {len(self.problems)} finding(s):"
        body = [f"  - {p.summary()}" for p in sorted(self.problems, key=lambda p: -p.score)]
        return "\n".join([head, *body])


# --------------------------------------------------------------------------- #
# Ground-truth adapter, so the visualiser can draw labels too
# --------------------------------------------------------------------------- #

def report_from_record(record: dict) -> StudyReport:
    """Build a :class:`StudyReport` from a detectron2 dataset dict (ground truth).

    Lets the visualiser draw annotations and predictions through the same code
    path -- useful for side-by-side figures in the report.  Annotations that
    carry a diagnosis become problems; annotations that only carry a tooth id
    become teeth.
    """
    report = StudyReport(
        image_path=record["file_name"],
        width=record["width"],
        height=record["height"],
    )
    for ann in record.get("annotations", []):
        x, y, w, h = ann["bbox"]
        box: BBox = (x, y, x + w, y + h)
        tooth_id = ann.get("tooth_id", -1)
        tooth_id = tooth_id if tooth_id >= 0 else None
        disease_id = ann.get("disease_id", -1)
        if disease_id >= 0:
            report.problems.append(
                ProblemFinding(
                    bbox=box,
                    score=1.0,
                    disease_id=disease_id,
                    tooth_id=tooth_id,
                    tooth_source="head" if tooth_id is not None else "none",
                    link_iou=1.0,
                    head_tooth_id=tooth_id,
                )
            )
        if tooth_id is not None:
            report.teeth.append(ToothFinding(bbox=box, score=1.0, tooth_id=tooth_id))
    return report


def aggregate_disease_counts(reports: Sequence[StudyReport]) -> Dict[str, int]:
    """Count findings per diagnosis across many studies."""
    counts = {name: 0 for name in fdi.DISEASE_CLASS_NAMES}
    for report in reports:
        for problem in report.problems:
            counts[problem.disease] += 1
    return counts


__all__ = [
    "BBox",
    "ProblemFinding",
    "StudyReport",
    "ToothFinding",
    "aggregate_disease_counts",
    "report_from_record",
]
