"""Evaluation: standard detection AP plus the metric DENTEX actually cares about.

``COCOEvaluator`` scores the primary head alone -- for the pathology model that
is "did you find a caries lesion somewhere", which says nothing about whether
the model named the right tooth.  DENTEX's own criterion is stricter: a
detection counts only when the quadrant, the enumeration *and* the diagnosis
are all right.

:class:`HierarchyEvaluator` adds that.  It greedily matches predictions to
ground truth by box IoU and then reports, over the matched pairs:

* ``detection/recall`` and ``detection/precision`` -- localisation alone;
* ``level/<name>`` -- accuracy of each hierarchy level on true positives;
* ``fdi/accuracy`` -- fraction of true positives whose full FDI number is right;
* ``triple/accuracy`` -- quadrant + position + diagnosis all correct, the
  DENTEX definition;
* ``triple/recall`` -- the same, as a share of all ground-truth instances,
  which is the single number worth quoting in the report.

The evaluator reads whichever levels a prediction carries, so it works for the
tooth model (primary ``tooth``) and the pathology model (primary ``disease``
with ``quadrant`` + ``position`` auxiliaries) without configuration.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils import comm

from .. import fdi
from ..modeling.hierarchy import TASK_PRIMARY_LEVEL

logger = logging.getLogger(__name__)


class HierarchyEvaluator(DatasetEvaluator):
    """Match predictions to ground truth by IoU and score the label hierarchy."""

    def __init__(
        self,
        dataset_name: str,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        distributed: bool = True,
    ):
        self.dataset_name = dataset_name
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.distributed = distributed

        metadata = MetadataCatalog.get(dataset_name)
        self.task = metadata.get("dentex_task", "disease")
        self.primary_level = TASK_PRIMARY_LEVEL[self.task]
        self._ground_truth = {
            record["image_id"]: record for record in DatasetCatalog.get(dataset_name)
        }
        self._predictions: List[dict] = []

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._predictions = []

    def process(self, inputs: Sequence[dict], outputs: Sequence[dict]) -> None:
        for input_dict, output in zip(inputs, outputs):
            instances = output.get("instances")
            if instances is None:
                continue
            instances = instances.to("cpu")
            keep = instances.scores >= self.score_threshold
            instances = instances[keep]

            prediction = {
                "image_id": input_dict["image_id"],
                "boxes": instances.pred_boxes.tensor.numpy(),
                "scores": instances.scores.numpy(),
                self.primary_level: instances.pred_classes.numpy(),
            }
            for level in ("quadrant", "position", "tooth", "disease"):
                field = f"pred_{level}"
                if level != self.primary_level and instances.has(field):
                    prediction[level] = instances.get(field).numpy()
            self._predictions.append(prediction)

    # ------------------------------------------------------------------ #

    def evaluate(self) -> Optional[OrderedDict]:
        predictions = self._predictions
        if self.distributed:
            comm.synchronize()
            gathered = comm.gather(predictions, dst=0)
            if not comm.is_main_process():
                return None
            predictions = list(itertools.chain(*gathered))

        if not predictions:
            logger.warning("HierarchyEvaluator received no predictions")
            return OrderedDict()

        n_gt = 0
        n_pred = 0
        n_matched = 0
        level_correct: Dict[str, int] = defaultdict(int)
        level_total: Dict[str, int] = defaultdict(int)
        fdi_correct = 0
        fdi_total = 0
        triple_correct = 0
        confusion: Counter = Counter()

        for prediction in predictions:
            record = self._ground_truth.get(prediction["image_id"])
            if record is None:
                continue
            gt_annotations = record["annotations"]
            n_gt += len(gt_annotations)
            n_pred += len(prediction["scores"])
            if not gt_annotations or not len(prediction["scores"]):
                continue

            gt_boxes = Boxes(
                torch.tensor(
                    [
                        [a["bbox"][0], a["bbox"][1],
                         a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                        for a in gt_annotations
                    ],
                    dtype=torch.float32,
                )
            )
            pred_boxes = Boxes(torch.as_tensor(prediction["boxes"], dtype=torch.float32))
            iou = pairwise_iou(pred_boxes, gt_boxes).numpy()

            order = np.argsort(-prediction["scores"])
            taken = set()
            for pred_index in order:
                candidates = iou[pred_index].copy()
                for gt_index in taken:
                    candidates[gt_index] = -1.0
                best = int(np.argmax(candidates))
                if candidates[best] < self.iou_threshold:
                    continue
                taken.add(best)
                n_matched += 1
                gt_annotation = gt_annotations[best]

                for level in ("quadrant", "position", "tooth", "disease"):
                    if level not in prediction:
                        continue
                    truth = gt_annotation.get(f"{level}_id", -1)
                    if truth < 0:
                        continue
                    level_total[level] += 1
                    if int(prediction[level][pred_index]) == int(truth):
                        level_correct[level] += 1

                predicted_tooth = self._predicted_tooth(prediction, pred_index)
                true_tooth = gt_annotation.get("tooth_id", -1)
                if true_tooth >= 0 and predicted_tooth is not None:
                    fdi_total += 1
                    tooth_ok = predicted_tooth == true_tooth
                    fdi_correct += int(tooth_ok)
                    disease_ok = True
                    if "disease" in prediction and gt_annotation.get("disease_id", -1) >= 0:
                        disease_ok = int(prediction["disease"][pred_index]) == int(
                            gt_annotation["disease_id"]
                        )
                    triple_correct += int(tooth_ok and disease_ok)
                    if not tooth_ok:
                        confusion[
                            (
                                fdi.fdi_from_tooth_class_id(true_tooth),
                                fdi.fdi_from_tooth_class_id(predicted_tooth),
                            )
                        ] += 1

        results = OrderedDict()
        results["detection/recall"] = 100.0 * n_matched / max(n_gt, 1)
        results["detection/precision"] = 100.0 * n_matched / max(n_pred, 1)
        results["detection/n_gt"] = float(n_gt)
        results["detection/n_pred"] = float(n_pred)
        for level, total in sorted(level_total.items()):
            results[f"level/{level}"] = 100.0 * level_correct[level] / max(total, 1)
        if fdi_total:
            results["fdi/accuracy"] = 100.0 * fdi_correct / fdi_total
            results["triple/accuracy"] = 100.0 * triple_correct / fdi_total
            results["triple/recall"] = 100.0 * triple_correct / max(n_gt, 1)

        if confusion:
            worst = ", ".join(
                f"{true}->{pred} x{count}"
                for (true, pred), count in confusion.most_common(5)
            )
            logger.info("[HierarchyEvaluator] most frequent enumeration errors: %s", worst)

        logger.info("[HierarchyEvaluator] %s", dict(results))
        return OrderedDict({"dentex_hierarchy": results})

    # ------------------------------------------------------------------ #

    @staticmethod
    def _predicted_tooth(prediction: dict, index: int) -> Optional[int]:
        """Resolve a tooth id from whichever heads the model has."""
        if "tooth" in prediction:
            return int(prediction["tooth"][index])
        if "quadrant" in prediction and "position" in prediction:
            return fdi.tooth_class_id(
                int(prediction["quadrant"][index]) + 1,
                int(prediction["position"][index]) + 1,
            )
        return None


def build_evaluator(cfg, dataset_name: str, output_folder: Optional[str] = None):
    """COCO AP (boxes + masks) together with the hierarchical metrics."""
    from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

    evaluators = [
        COCOEvaluator(
            dataset_name,
            tasks=("bbox", "segm"),
            distributed=True,
            output_dir=output_folder,
            allow_cached_coco=False,
        ),
        HierarchyEvaluator(
            dataset_name,
            iou_threshold=0.5,
            score_threshold=cfg.MODEL.DENTEX.EVAL_SCORE_THRESHOLD,
        ),
    ]
    return DatasetEvaluators(evaluators)


__all__ = ["HierarchyEvaluator", "build_evaluator"]
