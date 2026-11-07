"""Abnormal-tooth detection on panoramic dental x-rays (DENTEX).

Two Mask2Former models, one hierarchy:

* ``dentex.data``       -- download DENTEX, convert it to detectron2 dicts,
                           register the splits, augment x-rays safely.
* ``dentex.modeling``   -- Mask2Former with box output and auxiliary
                           quadrant / position / diagnosis heads.
* ``dentex.engine``     -- trainer, multi-stage transfer, COCO AP plus the
                           hierarchical metric DENTEX is scored on.
* ``dentex.inference``  -- run both models, bind each finding to a tooth,
                           report it and draw it.

``dentex.fdi`` holds every label mapping, so the FDI numbering is defined once.

Only ``dentex.fdi``, ``dentex.data.convert`` and ``dentex.inference.report`` /
``visualize`` are importable without detectron2; everything else needs the
detectron2 + Mask2Former stack described in the README.
"""

from . import fdi

__version__ = "0.1.0"

__all__ = ["fdi", "__version__"]
