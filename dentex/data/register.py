"""Register the DENTEX splits with detectron2's ``DatasetCatalog``.

Dataset names follow ``dentex_<task>_<split>``::

    dentex_quadrant_train   dentex_quadrant_val      # 693 x-rays, warm-up stage
    dentex_tooth_train      dentex_tooth_val         # 634 x-rays, all teeth, 32 FDI classes
    dentex_disease_train    dentex_disease_val       # 705 x-rays, abnormal teeth, 4 diagnoses
    dentex_disease_holdout                           # the official 50 validation x-rays

``*_train`` / ``*_val`` come from a stable hash split of the released training
subsets (see :func:`dentex.data.convert.assign_splits`), so model selection
never touches ``dentex_disease_holdout``.  That hold-out is the official DENTEX
validation release, whose labels were published after the challenge closed, and
it is the split to quote in the report.

Registration is lazy: the json is only read when a dataset is first used, so
registering a split whose images were not downloaded costs nothing until you
actually ask for it.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from .. import fdi
from . import download
from .convert import TASK_CLASS_NAMES, TASK_NUM_CLASSES, load_dentex

logger = logging.getLogger(__name__)

#: ``task -> subset directory key`` in :data:`dentex.data.download.SUBSET_DIRS`.
TASK_TO_SUBSET = {
    "quadrant": "quadrant",
    "tooth": "enumeration",
    "disease": "disease",
}

DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SALT = "dentex"


def _metadata_for(task: str) -> dict:
    """Metadata every DENTEX dataset carries, so downstream code can label plots."""
    return {
        "thing_classes": list(TASK_CLASS_NAMES[task]),
        "dentex_task": task,
        "dentex_num_classes": TASK_NUM_CLASSES[task],
        "quadrant_classes": list(fdi.QUADRANT_NAMES),
        "position_classes": list(fdi.POSITION_NAMES),
        "tooth_classes": list(fdi.TOOTH_CLASS_NAMES),
        "disease_classes": list(fdi.DISEASE_CLASS_NAMES),
        "evaluator_type": "dentex",
    }


def register_split(
    name: str,
    annotation_json: str | Path,
    images_dir: str | Path,
    task: str,
    split: Optional[str],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    salt: str = DEFAULT_SALT,
    extra_metadata: Optional[dict] = None,
) -> str:
    """Register one split and return its name."""
    from detectron2.data import DatasetCatalog, MetadataCatalog

    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)

    DatasetCatalog.register(
        name,
        partial(
            load_dentex,
            annotation_json=str(annotation_json),
            images_dir=str(images_dir),
            task=task,
            split=split,
            val_fraction=val_fraction,
            salt=salt,
        ),
    )
    meta = _metadata_for(task)
    meta.update(
        {
            # Deliberately NOT "json_file": COCOEvaluator treats that key as a
            # ready-made COCO file and hands it straight to pycocotools, which
            # chokes on DENTEX's category_id_1/2/3 layout.  Leaving it unset
            # makes the evaluator convert the registered dicts itself.
            "dentex_json": str(annotation_json),
            "image_root": str(images_dir),
            "dentex_split": split or "all",
        }
    )
    if extra_metadata:
        meta.update(extra_metadata)
    MetadataCatalog.get(name).set(**meta)
    return name


def register_dentex(
    root: str | Path,
    tasks: Optional[List[str]] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    salt: str = DEFAULT_SALT,
    holdout_root: Optional[str | Path] = None,
    prefix: str = "dentex",
) -> Dict[str, str]:
    """Register every DENTEX split that is present under ``root``.

    Args:
        root: directory that :func:`dentex.data.download.fetch_training_data`
            wrote ``training_data/`` into.
        tasks: subset of ``{"quadrant", "tooth", "disease"}``; defaults to all.
        val_fraction: share of each released training subset held out for
            model selection.
        salt: split salt; keep fixed across experiments.
        holdout_root: directory that :func:`dentex.data.download.fetch_validation_data`
            wrote; registers ``<prefix>_disease_holdout`` when given.
        prefix: dataset-name prefix.

    Returns:
        ``{registered_name: description}`` for everything that was found.  Splits
        whose files are missing are skipped with a warning rather than raising,
        so a partial download still gives you a usable catalog.
    """
    root = Path(root)
    tasks = list(tasks or TASK_TO_SUBSET)
    registered: Dict[str, str] = {}

    for task in tasks:
        if task not in TASK_TO_SUBSET:
            raise ValueError(f"unknown task {task!r}; choose from {sorted(TASK_TO_SUBSET)}")
        try:
            annotation_json, images_dir = download.locate(root, TASK_TO_SUBSET[task])
        except FileNotFoundError as exc:
            logger.warning("skipping task %r: %s", task, exc)
            continue
        for split in ("train", "val"):
            name = f"{prefix}_{task}_{split}"
            register_split(
                name, annotation_json, images_dir, task, split, val_fraction, salt
            )
            registered[name] = f"{task}/{split} from {annotation_json.name}"

    if holdout_root is not None:
        try:
            gt_json, images_dir = download.find_validation(holdout_root)
        except FileNotFoundError as exc:
            logger.warning("skipping hold-out: %s", exc)
        else:
            name = f"{prefix}_disease_holdout"
            register_split(
                name, gt_json, images_dir, "disease", None, val_fraction, salt,
                extra_metadata={"dentex_split": "official_validation"},
            )
            registered[name] = "official DENTEX validation release (50 x-rays)"

            # The same 50 x-rays also carry quadrant + enumeration labels, which
            # makes them a usable (if small) hold-out for the tooth detector --
            # but only for the abnormal teeth that are annotated there.
            name = f"{prefix}_tooth_holdout_abnormal"
            register_split(
                name, gt_json, images_dir, "tooth", None, val_fraction, salt,
                extra_metadata={
                    "dentex_split": "official_validation",
                    "note": "only abnormal teeth are annotated; not a full tooth benchmark",
                },
            )
            registered[name] = "official validation, tooth labels (abnormal teeth only)"

    if not registered:
        logger.warning("no DENTEX splits were registered -- is %s the right root?", root)
    for name, description in registered.items():
        logger.info("registered %s: %s", name, description)
    return registered


__all__ = ["TASK_TO_SUBSET", "register_dentex", "register_split"]
