"""Data layer: download DENTEX, convert it, register it, augment it.

``download`` and ``convert`` have no detectron2 dependency, so the annotations
can be inspected from a plain Python session; ``register`` and ``mapper`` do.
"""

from .convert import (
    TASK_CLASS_NAMES,
    TASK_NUM_CLASSES,
    TASKS,
    assign_splits,
    dataset_statistics,
    load_dentex,
    to_coco_json,
)
from .download import (
    DEFAULT_SUBSETS,
    SUBSET_DIRS,
    fetch_test_data,
    fetch_training_data,
    fetch_validation_data,
    find_validation,
    locate,
)

__all__ = [
    "DEFAULT_SUBSETS",
    "SUBSET_DIRS",
    "TASKS",
    "TASK_CLASS_NAMES",
    "TASK_NUM_CLASSES",
    "assign_splits",
    "dataset_statistics",
    "fetch_test_data",
    "fetch_training_data",
    "fetch_validation_data",
    "find_validation",
    "load_dentex",
    "locate",
    "to_coco_json",
]

try:  # pragma: no cover - needs detectron2
    from .mapper import DentexDatasetMapper, build_augmentations, mirror_annotation
    from .register import register_dentex, register_split

    __all__ += [
        "DentexDatasetMapper",
        "build_augmentations",
        "mirror_annotation",
        "register_dentex",
        "register_split",
    ]
except ImportError:  # detectron2 missing
    pass
