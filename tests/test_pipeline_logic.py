"""Tests for the parts that do not need detectron2.

Run with ``python -m pytest tests -q`` (or just ``python tests/test_pipeline_logic.py``).
These cover the logic most likely to be silently wrong: the FDI mappings, the
DENTEX category-id remap, the mirror-aware flip, deduplication and the
tooth/pathology linking rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dentex import fdi  # noqa: E402
from dentex.inference.pipeline import (  # noqa: E402
    DentexPipeline,
    RawInstances,
    box_iou,
    deduplicate,
)
from dentex.inference.report import StudyReport, ToothFinding  # noqa: E402


# --------------------------------------------------------------------------- #
# FDI
# --------------------------------------------------------------------------- #

def test_fdi_roundtrip():
    for quadrant in fdi.QUADRANTS:
        for position in fdi.POSITIONS:
            number = fdi.quadrant_position_to_fdi(quadrant, position)
            assert fdi.fdi_to_quadrant_position(number) == (quadrant, position)
            class_id = fdi.tooth_class_id(quadrant, position)
            assert fdi.fdi_from_tooth_class_id(class_id) == number
            assert fdi.split_tooth_class_id(class_id) == (quadrant, position)
    assert len(set(fdi.FDI_NUMBERS)) == fdi.NUM_TEETH == 32


def test_category_remap_resolves_by_name_not_id():
    # train_quadrant.json really does ship ids out of order.
    categories = [
        {"id": 0, "name": "2"},
        {"id": 1, "name": "1"},
        {"id": 2, "name": "3"},
        {"id": 3, "name": "4"},
    ]
    assert fdi.quadrant_remap(categories) == {0: 1, 1: 0, 2: 2, 3: 3}


def test_category_remap_rejects_unexpected_names():
    try:
        fdi.disease_remap([{"id": 0, "name": "Cavity"}])
    except ValueError:
        return
    raise AssertionError("expected a ValueError for an unknown diagnosis name")


# --------------------------------------------------------------------------- #
# Mirror-aware flip
# --------------------------------------------------------------------------- #

def test_mirror_swaps_quadrants_and_keeps_position():
    from dentex.data.mapper import mirror_annotation

    # Tooth 18 -> quadrant 1, position 8 -> mirrored to quadrant 2 -> tooth 28.
    annotation = {
        "quadrant_id": 0,
        "position_id": 7,
        "tooth_id": fdi.tooth_class_id(1, 8),
        "disease_id": 1,
        "category_id": fdi.tooth_class_id(1, 8),
    }
    mirrored = mirror_annotation(dict(annotation), task="tooth")
    assert mirrored["quadrant_id"] == 1
    assert mirrored["position_id"] == 7
    assert fdi.fdi_from_tooth_class_id(mirrored["tooth_id"]) == 28
    assert mirrored["category_id"] == mirrored["tooth_id"]

    # Mirroring twice is the identity.
    assert mirror_annotation(mirrored, task="tooth")["tooth_id"] == annotation["tooth_id"]

    # On the disease task the primary label is the diagnosis, untouched.
    again = mirror_annotation(dict(annotation), task="disease")
    assert again["category_id"] == 1


# --------------------------------------------------------------------------- #
# Geometry and deduplication
# --------------------------------------------------------------------------- #

def test_box_iou():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    iou = box_iou(a, b)[0]
    assert np.isclose(iou[0], 1.0)
    assert np.isclose(iou[1], 50 / 150)
    assert np.isclose(iou[2], 0.0)


def _raw(boxes, scores, classes, levels=None):
    return RawInstances(
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        classes=np.asarray(classes, dtype=np.int64),
        levels={k: np.asarray(v, dtype=np.int64) for k, v in (levels or {}).items()} or None,
    )


def test_deduplicate_drops_low_scores_and_overlaps():
    raw = _raw(
        boxes=[[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]],
        scores=[0.9, 0.8, 0.2],
        classes=[5, 5, 7],
    )
    kept = deduplicate(raw, score_threshold=0.5, nms_threshold=0.6)
    assert len(kept) == 1
    assert kept.scores[0] == np.float32(0.9)


def test_deduplicate_unique_per_class_keeps_one_tooth_per_fdi():
    # Same FDI number predicted twice in different places: anatomically only
    # one can be right, so the lower-scoring one must go.
    raw = _raw(
        boxes=[[0, 0, 10, 10], [500, 0, 510, 10], [20, 0, 30, 10]],
        scores=[0.9, 0.7, 0.8],
        classes=[5, 5, 6],
    )
    kept = deduplicate(raw, score_threshold=0.5, nms_threshold=0.6, unique_per_class=True)
    assert sorted(kept.classes.tolist()) == [5, 6]
    assert kept.boxes[kept.classes.tolist().index(5)].tolist() == [0, 0, 10, 10]


# --------------------------------------------------------------------------- #
# Linking a finding to a tooth
# --------------------------------------------------------------------------- #

def _pipeline():
    return DentexPipeline(
        tooth_predictor=object(),      # never called in these tests
        disease_predictor=object(),
        link_iou_threshold=0.3,
    )


def test_link_prefers_tooth_detector_over_its_own_head():
    tooth_36 = fdi.tooth_class_id(3, 6)
    tooth_37 = fdi.tooth_class_id(3, 7)
    teeth = _raw([[0, 0, 100, 200]], [0.95], [tooth_36])
    problems = _raw(
        [[5, 5, 95, 195]],
        [0.8],
        [1],  # Caries
        levels={"quadrant": [2], "position": [6]},  # the head says tooth 37
    )
    [finding] = _pipeline().link(problems, teeth)
    assert finding.tooth_id == tooth_36
    assert finding.head_tooth_id == tooth_37
    assert finding.tooth_source == "linked"
    assert finding.link_iou > 0.8
    assert finding.disease == "Caries"


def test_link_marks_agreement():
    tooth_36 = fdi.tooth_class_id(3, 6)
    teeth = _raw([[0, 0, 100, 200]], [0.95], [tooth_36])
    problems = _raw([[5, 5, 95, 195]], [0.8], [1], levels={"quadrant": [2], "position": [5]})
    [finding] = _pipeline().link(problems, teeth)
    assert finding.tooth_id == tooth_36
    assert finding.tooth_source == "agreed"


def test_link_falls_back_to_the_head_when_nothing_overlaps():
    teeth = _raw([[1000, 1000, 1100, 1200]], [0.95], [0])
    problems = _raw([[0, 0, 100, 200]], [0.8], [3], levels={"quadrant": [3], "position": [7]})
    [finding] = _pipeline().link(problems, teeth)
    assert finding.tooth_source == "head"
    assert finding.fdi == 48
    assert finding.disease == "Deep Caries"


def test_link_without_any_tooth_information():
    problems = _raw([[0, 0, 100, 200]], [0.8], [0])
    [finding] = _pipeline().link(problems, None)
    assert finding.tooth_id is None
    assert finding.tooth_source == "none"
    assert "unmatched" in finding.summary()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def test_report_groups_problems_by_tooth():
    report = StudyReport(image_path="x.png", width=100, height=100)
    report.teeth.append(ToothFinding(bbox=(0, 0, 10, 10), score=0.9, tooth_id=0))
    problems = _raw(
        [[0, 0, 10, 10], [0, 0, 10, 10]],
        [0.9, 0.7],
        [1, 3],
        levels={"quadrant": [0], "position": [0]},
    )
    # same tooth, two diagnoses
    problems.levels = {"quadrant": np.array([0, 0]), "position": np.array([0, 0])}
    report.problems = _pipeline().link(problems, None)
    grouped = report.problems_by_tooth()
    assert list(grouped) == [11]
    assert [p.disease for p in grouped[11]] == ["Caries", "Deep Caries"]
    assert "| 11 | Caries |" in report.to_markdown()


def _run_all() -> int:
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"ok   {name}")
    print(f"\n{failures} failure(s)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
