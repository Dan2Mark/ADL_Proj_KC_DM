"""Inference: run the models, link findings to teeth, report and draw them."""

from .pipeline import DentexPipeline, RawInstances, box_iou, deduplicate, instances_to_raw
from .report import (
    ProblemFinding,
    StudyReport,
    ToothFinding,
    aggregate_disease_counts,
    report_from_record,
)
from .visualize import (
    DISEASE_COLORS,
    TOOTH_COLOR,
    contact_sheet,
    draw_report,
    load_xray,
    polygons_to_mask,
    side_by_side,
)

__all__ = [
    "DISEASE_COLORS",
    "DentexPipeline",
    "ProblemFinding",
    "RawInstances",
    "StudyReport",
    "TOOTH_COLOR",
    "ToothFinding",
    "aggregate_disease_counts",
    "box_iou",
    "contact_sheet",
    "deduplicate",
    "draw_report",
    "instances_to_raw",
    "load_xray",
    "polygons_to_mask",
    "report_from_record",
    "side_by_side",
]
