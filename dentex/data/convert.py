"""Turn DENTEX json into detectron2 dataset dicts.

DENTEX is *almost* COCO: the images list is standard, annotations carry ``bbox``
(XYWH, absolute), ``area``, ``iscrowd`` and a single-part ``segmentation``
polygon, but the single ``category_id`` is replaced by up to three parallel ids
(``category_id_1/2/3``) for the quadrant / position / diagnosis levels.  That is
why the stock ``load_coco_json`` cannot be used directly.

Three tasks are supported, matching the three annotated subsets:

``quadrant``
    693 x-rays, 4 classes, one box per jaw quadrant.  Used only as a warm-up
    stage -- the objects are jaw regions, not teeth.

``tooth``
    634 x-rays, 32 classes (FDI 11..48), **every** tooth annotated
    (28.5 per image).  This trains the tooth detector that produces the green
    boxes with FDI numbers.

``disease``
    705 x-rays, 4 diagnosis classes, **only abnormal teeth** annotated
    (5.0 per image), each also carrying its quadrant and position.  This trains
    the hierarchical pathology model that produces the red boxes.

Because the disease subset leaves healthy teeth unlabelled, the two detectors
must stay separate: training tooth detection on the disease subset would teach
the model to treat healthy teeth as background.  See the project README.

Every emitted annotation carries all hierarchy levels it knows about, with
``-1`` for levels the subset does not label::

    {"bbox": [x, y, w, h], "bbox_mode": BoxMode.XYWH_ABS,
     "segmentation": [[x1, y1, x2, y2, ...]],
     "category_id":  <primary label for the chosen task>,
     "quadrant_id":  0..3   or -1,
     "position_id":  0..7   or -1,
     "tooth_id":     0..31  or -1,
     "disease_id":   0..3   or -1,
     "iscrowd": 0}
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import fdi

logger = logging.getLogger(__name__)

TASKS = ("quadrant", "tooth", "disease")

try:  # detectron2 is optional here, so the annotations stay inspectable
    from detectron2.structures import BoxMode as _BoxMode

    XYWH_ABS = _BoxMode.XYWH_ABS
except ImportError:  # pragma: no cover
    #: ``BoxMode.XYWH_ABS``.  ``BoxMode`` is an ``IntEnum``, so the plain int
    #: compares equal to it and detectron2 accepts either.
    XYWH_ABS = 1

#: Number of classes in the primary (matched) head for each task.
TASK_NUM_CLASSES = {
    "quadrant": fdi.NUM_QUADRANTS,
    "tooth": fdi.NUM_TEETH,
    "disease": fdi.NUM_DISEASES,
}

#: Primary class names for each task, in contiguous-id order.
TASK_CLASS_NAMES = {
    "quadrant": list(fdi.QUADRANT_NAMES),
    "tooth": list(fdi.TOOTH_CLASS_NAMES),
    "disease": list(fdi.DISEASE_CLASS_NAMES),
}


# --------------------------------------------------------------------------- #
# Deterministic splitting
# --------------------------------------------------------------------------- #

def _split_bucket(file_name: str, salt: str) -> float:
    """Stable pseudo-random number in [0, 1) derived from the image file name.

    Hashing the file name rather than shuffling a list keeps the split identical
    across machines, Python versions and reruns, and keeps it stable if the
    subset ever grows.
    """
    digest = hashlib.sha256(f"{salt}/{file_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_splits(
    file_names: Sequence[str],
    val_fraction: float = 0.1,
    salt: str = "dentex",
) -> Dict[str, str]:
    """Map each image file name to ``"train"`` or ``"val"``."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    return {
        name: ("val" if _split_bucket(name, salt) < val_fraction else "train")
        for name in file_names
    }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _remaps(raw: dict, task: str) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Build ``(quadrant, position, disease)`` id remaps for a DENTEX json.

    Handles both layouts: the quadrant-only file uses a single ``categories``
    key, the richer files use ``categories_1/2/3``.
    """
    if "categories_1" in raw:
        q = fdi.quadrant_remap(raw["categories_1"])
        p = fdi.position_remap(raw["categories_2"]) if "categories_2" in raw else None
        d = fdi.disease_remap(raw["categories_3"]) if "categories_3" in raw else None
    elif "categories" in raw:
        q, p, d = fdi.quadrant_remap(raw["categories"]), None, None
    else:
        raise ValueError("DENTEX json has neither 'categories' nor 'categories_1'")

    if task == "tooth" and p is None:
        raise ValueError("task 'tooth' needs a json with 'categories_2' (the enumeration subset)")
    if task == "disease" and d is None:
        raise ValueError("task 'disease' needs a json with 'categories_3' (the disease subset)")
    return q, p, d


def _annotation_levels(
    ann: dict,
    q_map: Optional[dict],
    p_map: Optional[dict],
    d_map: Optional[dict],
) -> Tuple[int, int, int, int]:
    """Resolve one annotation into ``(quadrant_id, position_id, tooth_id, disease_id)``."""
    if "category_id_1" in ann:
        raw_q = ann["category_id_1"]
    elif "category_id" in ann:
        raw_q = ann["category_id"]
    else:
        raise ValueError(f"annotation {ann.get('id')} has no quadrant id")
    quadrant_id = q_map[int(raw_q)] if q_map is not None else -1

    position_id = -1
    if p_map is not None and "category_id_2" in ann:
        position_id = p_map[int(ann["category_id_2"])]

    tooth_id = -1
    if quadrant_id >= 0 and position_id >= 0:
        tooth_id = fdi.tooth_class_id(quadrant_id + 1, position_id + 1)

    disease_id = -1
    if d_map is not None and "category_id_3" in ann:
        disease_id = d_map[int(ann["category_id_3"])]

    return quadrant_id, position_id, tooth_id, disease_id


def _valid_polygon(segmentation) -> Optional[List[List[float]]]:
    """Keep only polygon parts with at least three vertices; ``None`` if nothing is left."""
    if not isinstance(segmentation, list):
        return None
    parts = []
    for part in segmentation:
        if isinstance(part, list) and len(part) >= 6 and len(part) % 2 == 0:
            parts.append([float(v) for v in part])
    return parts or None


def load_dentex(
    annotation_json: str | Path,
    images_dir: str | Path,
    task: str,
    split: Optional[str] = None,
    val_fraction: float = 0.1,
    salt: str = "dentex",
    require_images: bool = True,
    check_images: bool = True,
) -> List[dict]:
    """Load a DENTEX subset as a list of detectron2 dataset dicts.

    Args:
        annotation_json: path to ``train_*.json`` or ``validation_triple.json``.
        images_dir: directory holding the ``.png`` x-rays named in the json.
        task: one of :data:`TASKS`; decides what ``category_id`` means.
        split: ``"train"``, ``"val"``, or ``None`` to return every image.
        val_fraction: share of images routed to ``"val"`` by the stable hash split.
        salt: changes the split; keep it fixed to keep results comparable.
        require_images: raise if an image named in the json is missing on disk;
            when ``False``, such images are dropped with a warning instead.
        check_images: set to ``False`` to skip touching the filesystem entirely
            and keep every record.  Only useful for offline inspection of the
            annotations (statistics, class balance) -- the returned
            ``file_name`` values may not exist.

    Returns:
        One dict per image, in file-name order.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    annotation_json = Path(annotation_json)
    images_dir = Path(images_dir)
    raw = _read_json(annotation_json)
    q_map, p_map, d_map = _remaps(raw, task)

    images = {int(img["id"]): img for img in raw["images"]}
    by_image: Dict[int, List[dict]] = {image_id: [] for image_id in images}
    dropped_no_poly = 0
    dropped_no_label = 0

    for ann in raw["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in by_image:
            logger.warning("annotation %s points at unknown image %s", ann.get("id"), image_id)
            continue

        polygon = _valid_polygon(ann.get("segmentation"))
        if polygon is None:
            dropped_no_poly += 1
            continue

        quadrant_id, position_id, tooth_id, disease_id = _annotation_levels(
            ann, q_map, p_map, d_map
        )
        primary = {"quadrant": quadrant_id, "tooth": tooth_id, "disease": disease_id}[task]
        if primary < 0:
            dropped_no_label += 1
            continue

        x, y, w, h = (float(v) for v in ann["bbox"])
        by_image[image_id].append(
            {
                "bbox": [x, y, w, h],
                "bbox_mode": XYWH_ABS,
                "segmentation": polygon,
                "category_id": int(primary),
                "quadrant_id": int(quadrant_id),
                "position_id": int(position_id),
                "tooth_id": int(tooth_id),
                "disease_id": int(disease_id),
                "area": float(ann.get("area", w * h)),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
        )

    if dropped_no_poly:
        logger.warning("%s: dropped %d annotation(s) with an unusable polygon",
                       annotation_json.name, dropped_no_poly)
    if dropped_no_label:
        logger.warning("%s: dropped %d annotation(s) with no %s label",
                       annotation_json.name, dropped_no_label, task)

    file_names = sorted(img["file_name"] for img in images.values())
    splits = assign_splits(file_names, val_fraction=val_fraction, salt=salt)

    records: List[dict] = []
    missing: List[str] = []
    for image_id, img in sorted(images.items(), key=lambda kv: kv[1]["file_name"]):
        file_name = img["file_name"]
        if split is not None and splits[file_name] != split:
            continue
        path = images_dir / file_name
        if check_images and not path.is_file():
            missing.append(file_name)
            continue
        records.append(
            {
                "file_name": str(path),
                "image_id": image_id,
                "height": int(img["height"]),
                "width": int(img["width"]),
                "dentex_split": splits[file_name],
                "annotations": by_image[image_id],
            }
        )

    if missing:
        message = (
            f"{len(missing)} image(s) named in {annotation_json.name} are not in "
            f"{images_dir} (first few: {missing[:5]})"
        )
        if require_images:
            raise FileNotFoundError(message)
        logger.warning(message)

    logger.info(
        "loaded %s task=%s split=%s: %d images, %d annotations",
        annotation_json.name, task, split, len(records),
        sum(len(r["annotations"]) for r in records),
    )
    return records


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def dataset_statistics(records: Sequence[dict]) -> dict:
    """Summarise a loaded split -- handy for the project report.

    Returns counts of images and instances, instances per image, and the class
    histogram at every hierarchy level that the split actually labels.
    """
    n_images = len(records)
    instances = [ann for rec in records for ann in rec["annotations"]]
    quadrants = Counter(a["quadrant_id"] for a in instances if a["quadrant_id"] >= 0)
    positions = Counter(a["position_id"] for a in instances if a["position_id"] >= 0)
    teeth = Counter(a["tooth_id"] for a in instances if a["tooth_id"] >= 0)
    diseases = Counter(a["disease_id"] for a in instances if a["disease_id"] >= 0)

    sizes = []
    for ann in instances:
        _, _, w, h = ann["bbox"]
        sizes.append(w * h)
    sizes.sort()

    return {
        "images": n_images,
        "instances": len(instances),
        "instances_per_image": round(len(instances) / max(n_images, 1), 2),
        "images_without_annotations": sum(1 for r in records if not r["annotations"]),
        "quadrants": {fdi.QUADRANT_NAMES[k]: v for k, v in sorted(quadrants.items())},
        "positions": {fdi.POSITION_NAMES[k]: v for k, v in sorted(positions.items())},
        "teeth": {fdi.fdi_from_tooth_class_id(k): v for k, v in sorted(teeth.items())},
        "diseases": {fdi.DISEASE_CLASS_NAMES[k]: v for k, v in sorted(diseases.items())},
        "median_box_area_px": int(sizes[len(sizes) // 2]) if sizes else 0,
    }


def to_coco_json(records: Sequence[dict], task: str, out_path: str | Path) -> Path:
    """Write a plain single-label COCO file for the chosen task.

    Useful for sanity-checking with external tools (``pycocotools``, FiftyOne,
    CVAT) and for anyone who wants to train a different detector on the same
    split without re-reading the DENTEX layout.
    """
    out_path = Path(out_path)
    images, annotations = [], []
    ann_id = 1
    for rec in records:
        images.append(
            {
                "id": rec["image_id"],
                "file_name": Path(rec["file_name"]).name,
                "height": rec["height"],
                "width": rec["width"],
            }
        )
        for ann in rec["annotations"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": rec["image_id"],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "quadrant_id": ann["quadrant_id"],
                    "position_id": ann["position_id"],
                    "tooth_id": ann["tooth_id"],
                    "disease_id": ann["disease_id"],
                }
            )
            ann_id += 1

    payload = {
        "info": {"description": f"DENTEX, task={task}"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name, "supercategory": task}
            for i, name in enumerate(TASK_CLASS_NAMES[task])
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return out_path


__all__ = [
    "TASKS",
    "TASK_CLASS_NAMES",
    "TASK_NUM_CLASSES",
    "assign_splits",
    "dataset_statistics",
    "load_dentex",
    "to_coco_json",
]
