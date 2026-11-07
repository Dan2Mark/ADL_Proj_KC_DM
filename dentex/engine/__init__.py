"""Training and evaluation."""

try:  # pragma: no cover - needs detectron2 + Mask2Former
    from .evaluator import HierarchyEvaluator, build_evaluator
    from .trainer import DentexTrainer, filter_checkpoint, inspect_transfer

    __all__ = [
        "DentexTrainer",
        "HierarchyEvaluator",
        "build_evaluator",
        "filter_checkpoint",
        "inspect_transfer",
    ]
except ImportError:  # pragma: no cover
    __all__ = []
