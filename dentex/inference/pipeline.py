"""End-to-end inference: teeth, pathology, and the link between them.

Two models run on the same x-ray.

* The **tooth** model is trained on the enumeration subset, where every tooth
  is annotated, so it finds all teeth and numbers them (FDI 11..48).
* The **pathology** model is trained on the diagnosis subset, where only
  abnormal teeth are annotated, so it finds the diseased ones and names the
  diagnosis -- and, through its auxiliary heads, its own guess at the quadrant
  and position.

They have to stay separate: training tooth detection on the diagnosis subset
would teach the model that healthy teeth are background, since they are left
unlabelled there.

That leaves two opinions about which tooth a finding sits on -- the pathology
model's own enumeration heads, and whichever tooth box overlaps the finding.
:meth:`DentexPipeline.link` reconciles them, which is worth doing: enumeration
is the hardest part of DENTEX, and the tooth model has seen 18k annotated teeth
against the pathology model's 3.5k.

Raw Mask2Former output needs cleaning before any of this.  It always returns
``TEST.DETECTIONS_PER_IMAGE`` query/class pairs, so the same tooth comes back
many times at different scores; :func:`deduplicate` applies a score threshold,
non-maximum suppression, and the anatomical constraint that an FDI number can
occur at most once per mouth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .. import fdi
from .report import ProblemFinding, StudyReport, ToothFinding

logger = logging.getLogger(__name__)

ImageLike = Union[str, Path, np.ndarray]


# --------------------------------------------------------------------------- #
# Post-processing of raw instances
# --------------------------------------------------------------------------- #

@dataclass
class RawInstances:
    """Model output reduced to plain arrays, so this module stays torch-light."""

    boxes: np.ndarray            # [N, 4] xyxy
    scores: np.ndarray           # [N]
    classes: np.ndarray          # [N] primary label
    masks: Optional[np.ndarray] = None   # [N, H, W] bool
    levels: Optional[Dict[str, np.ndarray]] = None   # auxiliary level labels
    level_scores: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def select(self, index) -> "RawInstances":
        return RawInstances(
            boxes=self.boxes[index],
            scores=self.scores[index],
            classes=self.classes[index],
            masks=None if self.masks is None else self.masks[index],
            levels={k: v[index] for k, v in (self.levels or {}).items()} or None,
            level_scores={k: v[index] for k, v in (self.level_scores or {}).items()} or None,
        )


def instances_to_raw(instances) -> RawInstances:
    """Convert a detectron2 ``Instances`` (already on CPU) to :class:`RawInstances`."""
    levels: Dict[str, np.ndarray] = {}
    level_scores: Dict[str, np.ndarray] = {}
    for name in ("quadrant", "position", "tooth", "disease"):
        if instances.has(f"pred_{name}"):
            levels[name] = instances.get(f"pred_{name}").numpy()
        if instances.has(f"pred_{name}_score"):
            level_scores[name] = instances.get(f"pred_{name}_score").numpy()
    return RawInstances(
        boxes=instances.pred_boxes.tensor.numpy(),
        scores=instances.scores.numpy(),
        classes=instances.pred_classes.numpy(),
        masks=instances.pred_masks.numpy().astype(bool) if instances.has("pred_masks") else None,
        levels=levels or None,
        level_scores=level_scores or None,
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``[N, 4]`` / ``[M, 4]`` xyxy box arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    overlap = np.clip(right_bottom - left_top, 0, None)
    intersection = overlap[..., 0] * overlap[..., 1]
    union = area_a[:, None] + area_b[None, :] - intersection
    return (intersection / np.maximum(union, 1e-6)).astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    """Plain greedy NMS; small N here, so NumPy is plenty."""
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][ious < threshold]
    return keep


def deduplicate(
    raw: RawInstances,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.6,
    unique_per_class: bool = False,
    max_instances: Optional[int] = None,
) -> RawInstances:
    """Turn the raw query/class grid into a clean set of detections.

    Args:
        raw: model output.
        score_threshold: drop anything below this.
        nms_threshold: greedy NMS IoU threshold, applied across all classes
            because two different FDI numbers should never cover one tooth.
        unique_per_class: keep only the best detection per class.  Correct for
            the tooth model (one tooth 36 per mouth) and wrong for the
            pathology model (several teeth can share a diagnosis).
        max_instances: optional hard cap after sorting by score.

    Returns:
        A filtered :class:`RawInstances`, ordered by descending score.
    """
    if len(raw) == 0:
        return raw

    keep = np.nonzero(raw.scores >= score_threshold)[0]
    if keep.size == 0:
        return raw.select(keep)
    raw = raw.select(keep)

    order = np.argsort(-raw.scores)
    raw = raw.select(order)

    if unique_per_class:
        seen: set[int] = set()
        unique: List[int] = []
        for i, label in enumerate(raw.classes):
            if int(label) in seen:
                continue
            seen.add(int(label))
            unique.append(i)
        raw = raw.select(np.asarray(unique, dtype=int))

    if nms_threshold < 1.0 and len(raw):
        raw = raw.select(np.asarray(_nms(raw.boxes, raw.scores, nms_threshold), dtype=int))

    if max_instances is not None and len(raw) > max_instances:
        raw = raw.select(np.arange(max_instances))
    return raw


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

class DentexPipeline:
    """Run the tooth and pathology models and link their outputs."""

    def __init__(
        self,
        tooth_predictor=None,
        disease_predictor=None,
        tooth_score_threshold: float = 0.5,
        disease_score_threshold: float = 0.5,
        tooth_nms_threshold: float = 0.6,
        disease_nms_threshold: float = 0.6,
        link_iou_threshold: float = 0.3,
        keep_masks: bool = True,
    ):
        if tooth_predictor is None and disease_predictor is None:
            raise ValueError("at least one of tooth_predictor / disease_predictor is required")
        self.tooth_predictor = tooth_predictor
        self.disease_predictor = disease_predictor
        self.tooth_score_threshold = tooth_score_threshold
        self.disease_score_threshold = disease_score_threshold
        self.tooth_nms_threshold = tooth_nms_threshold
        self.disease_nms_threshold = disease_nms_threshold
        self.link_iou_threshold = link_iou_threshold
        self.keep_masks = keep_masks

    # ------------------------------------------------------------------ #

    @classmethod
    def from_configs(
        cls,
        tooth_cfg=None,
        disease_cfg=None,
        **kwargs,
    ) -> "DentexPipeline":
        """Build both ``DefaultPredictor``s from frozen configs."""
        from detectron2.engine import DefaultPredictor

        import dentex.modeling  # noqa: F401  (registers the architecture)

        tooth_predictor = DefaultPredictor(tooth_cfg) if tooth_cfg is not None else None
        disease_predictor = DefaultPredictor(disease_cfg) if disease_cfg is not None else None
        return cls(tooth_predictor, disease_predictor, **kwargs)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _read(image: ImageLike) -> Tuple[np.ndarray, str]:
        if isinstance(image, np.ndarray):
            return image, "<array>"
        from detectron2.data.detection_utils import read_image

        path = str(image)
        return read_image(path, format="BGR"), path

    def __call__(self, image: ImageLike) -> StudyReport:
        """Analyse one x-ray and return the structured findings."""
        array, path = self._read(image)
        height, width = array.shape[:2]
        report = StudyReport(image_path=path, width=width, height=height)

        teeth_raw = self._run(
            self.tooth_predictor,
            array,
            self.tooth_score_threshold,
            self.tooth_nms_threshold,
            unique_per_class=True,
            max_instances=fdi.NUM_TEETH,
        )
        problems_raw = self._run(
            self.disease_predictor,
            array,
            self.disease_score_threshold,
            self.disease_nms_threshold,
            unique_per_class=False,
            max_instances=None,
        )

        if teeth_raw is not None:
            for i in range(len(teeth_raw)):
                report.teeth.append(
                    ToothFinding(
                        bbox=tuple(float(v) for v in teeth_raw.boxes[i]),
                        score=float(teeth_raw.scores[i]),
                        tooth_id=int(teeth_raw.classes[i]),
                        mask=(
                            teeth_raw.masks[i]
                            if self.keep_masks and teeth_raw.masks is not None
                            else None
                        ),
                    )
                )

        if problems_raw is not None:
            report.problems = self.link(problems_raw, teeth_raw)

        return report

    def _run(
        self,
        predictor,
        array: np.ndarray,
        score_threshold: float,
        nms_threshold: float,
        unique_per_class: bool,
        max_instances: Optional[int],
    ) -> Optional[RawInstances]:
        if predictor is None:
            return None
        with torch.no_grad():
            output = predictor(array)
        instances = output["instances"].to("cpu")
        return deduplicate(
            instances_to_raw(instances),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            unique_per_class=unique_per_class,
            max_instances=max_instances,
        )

    # ------------------------------------------------------------------ #

    def link(
        self,
        problems: RawInstances,
        teeth: Optional[RawInstances],
    ) -> List[ProblemFinding]:
        """Attach each pathology detection to a tooth.

        The pathology model's own quadrant/position heads give one answer; the
        overlapping tooth box gives another.  When both exist and agree, the
        finding is marked ``"agreed"``.  When they disagree, the tooth
        detector wins -- it was trained on 18k annotated teeth against the
        pathology model's 3.5k -- unless it has no overlapping box, in which
        case the model's own head is used and marked ``"head"``.
        """
        findings: List[ProblemFinding] = []
        if len(problems) == 0:
            return findings

        ious = (
            box_iou(problems.boxes, teeth.boxes)
            if teeth is not None and len(teeth)
            else np.zeros((len(problems), 0), dtype=np.float32)
        )

        for i in range(len(problems)):
            head_tooth = self._head_tooth(problems, i)

            matched_tooth: Optional[int] = None
            best_iou = 0.0
            if ious.shape[1]:
                best = int(np.argmax(ious[i]))
                best_iou = float(ious[i, best])
                if best_iou >= self.link_iou_threshold:
                    matched_tooth = int(teeth.classes[best])

            if matched_tooth is not None and head_tooth is not None:
                if matched_tooth == head_tooth:
                    tooth_id, source = matched_tooth, "agreed"
                else:
                    tooth_id, source = matched_tooth, "linked"
            elif matched_tooth is not None:
                tooth_id, source = matched_tooth, "linked"
            elif head_tooth is not None:
                tooth_id, source = head_tooth, "head"
            else:
                tooth_id, source = None, "none"

            findings.append(
                ProblemFinding(
                    bbox=tuple(float(v) for v in problems.boxes[i]),
                    score=float(problems.scores[i]),
                    disease_id=int(problems.classes[i]),
                    tooth_id=tooth_id,
                    tooth_source=source,
                    link_iou=best_iou,
                    head_tooth_id=head_tooth,
                    mask=(
                        problems.masks[i]
                        if self.keep_masks and problems.masks is not None
                        else None
                    ),
                )
            )
        return findings

    @staticmethod
    def _head_tooth(raw: RawInstances, index: int) -> Optional[int]:
        levels = raw.levels or {}
        if "tooth" in levels:
            return int(levels["tooth"][index])
        if "quadrant" in levels and "position" in levels:
            return fdi.tooth_class_id(
                int(levels["quadrant"][index]) + 1,
                int(levels["position"][index]) + 1,
            )
        return None

    # ------------------------------------------------------------------ #

    def run_many(self, images: Sequence[ImageLike]) -> List[StudyReport]:
        return [self(image) for image in images]


__all__ = [
    "DentexPipeline",
    "RawInstances",
    "box_iou",
    "deduplicate",
    "instances_to_raw",
]
