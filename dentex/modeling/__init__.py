"""Model components.

Importing this package registers the DENTEX meta-architecture and transformer
decoder with detectron2's registries, which is what makes
``MODEL.META_ARCHITECTURE: HierarchicalMaskFormer`` resolvable from a yaml file.
Mask2Former must already be importable (``sys.path`` pointing at a checkout, or
installed) -- see the README.
"""

from .hierarchy import LEVELS, TASK_PRIMARY_LEVEL, LevelSpec, get_level, resolve_levels

__all__ = [
    "LEVELS",
    "TASK_PRIMARY_LEVEL",
    "LevelSpec",
    "get_level",
    "resolve_levels",
]

try:  # pragma: no cover - depends on the optional detectron2/Mask2Former stack
    from .hier_criterion import HierarchicalSetCriterion, inverse_frequency_weights
    from .hier_decoder import HierarchicalMultiScaleMaskedTransformerDecoder
    from .hier_maskformer import HierarchicalMaskFormer

    __all__ += [
        "HierarchicalMaskFormer",
        "HierarchicalMultiScaleMaskedTransformerDecoder",
        "HierarchicalSetCriterion",
        "inverse_frequency_weights",
    ]
except ImportError as _exc:  # detectron2 / mask2former not installed
    import logging as _logging

    _logging.getLogger(__name__).debug(
        "dentex.modeling: heavy components unavailable (%s). "
        "The label hierarchy is still importable.",
        _exc,
    )
