"""Label spaces for DENTEX, centred on the FDI two-digit tooth notation.

DENTEX stores every annotation as a *hierarchy of three independent ids*::

    category_id_1 -> quadrant        (4 classes, FDI quadrants 1..4)
    category_id_2 -> position        (8 classes, tooth 1..8 inside the quadrant)
    category_id_3 -> diagnosis       (4 classes, only in the fully labelled subset)

An FDI number is ``quadrant * 10 + position`` -- e.g. quadrant 3, position 6 is
tooth 36 (lower-left first molar).  This module is the single place where the
mapping between raw DENTEX ids, contiguous training ids and FDI numbers lives,
so the data layer, the model heads, the evaluator and the visualiser can never
drift apart.

A trap worth knowing about: in ``train_quadrant.json`` the ``categories`` list is
*not* sorted by id (id 0 is named "2", id 1 is named "1").  Never assume that a
DENTEX category id equals its name minus one -- always resolve through the
``name`` field.  :func:`build_category_remap` does that.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Level 1: quadrants
# --------------------------------------------------------------------------- #

QUADRANTS: Tuple[int, ...] = (1, 2, 3, 4)
NUM_QUADRANTS = len(QUADRANTS)

QUADRANT_NAMES: Tuple[str, ...] = (
    "Q1 upper-right",
    "Q2 upper-left",
    "Q3 lower-left",
    "Q4 lower-right",
)

# --------------------------------------------------------------------------- #
# Level 2: position inside the quadrant
# --------------------------------------------------------------------------- #

POSITIONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
NUM_POSITIONS = len(POSITIONS)

POSITION_NAMES: Tuple[str, ...] = (
    "central incisor",
    "lateral incisor",
    "canine",
    "first premolar",
    "second premolar",
    "first molar",
    "second molar",
    "third molar",
)

# --------------------------------------------------------------------------- #
# Level 1 x Level 2: the 32 permanent teeth, used as the tooth detector's flat
# class space.  Contiguous id = (quadrant - 1) * 8 + (position - 1).
# --------------------------------------------------------------------------- #

NUM_TEETH = NUM_QUADRANTS * NUM_POSITIONS  # 32

FDI_NUMBERS: Tuple[int, ...] = tuple(
    q * 10 + p for q in QUADRANTS for p in POSITIONS
)
TOOTH_CLASS_NAMES: Tuple[str, ...] = tuple(str(n) for n in FDI_NUMBERS)

_FDI_TO_ID: Dict[int, int] = {fdi: i for i, fdi in enumerate(FDI_NUMBERS)}

# --------------------------------------------------------------------------- #
# Level 3: diagnosis.  Order matches DENTEX ``categories_3`` exactly.
# --------------------------------------------------------------------------- #

DISEASE_CLASS_NAMES: Tuple[str, ...] = (
    "Impacted",
    "Caries",
    "Periapical Lesion",
    "Deep Caries",
)
NUM_DISEASES = len(DISEASE_CLASS_NAMES)

#: Index of the implicit "no finding" label used by the per-tooth disease head.
HEALTHY_INDEX = NUM_DISEASES


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #

def quadrant_position_to_fdi(quadrant: int, position: int) -> int:
    """``(3, 6) -> 36``.  Arguments are 1-based, as printed in the FDI system."""
    if quadrant not in QUADRANTS:
        raise ValueError(f"quadrant must be one of {QUADRANTS}, got {quadrant}")
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position}")
    return quadrant * 10 + position


def fdi_to_quadrant_position(fdi: int) -> Tuple[int, int]:
    """``36 -> (3, 6)``."""
    quadrant, position = divmod(int(fdi), 10)
    if quadrant not in QUADRANTS or position not in POSITIONS:
        raise ValueError(f"{fdi} is not a valid permanent-tooth FDI number")
    return quadrant, position


def tooth_class_id(quadrant: int, position: int) -> int:
    """``(3, 6) -> 21``, a contiguous training id in ``[0, 32)``."""
    return (quadrant - 1) * NUM_POSITIONS + (position - 1)


def tooth_class_id_from_fdi(fdi: int) -> int:
    try:
        return _FDI_TO_ID[int(fdi)]
    except KeyError as exc:
        raise ValueError(f"{fdi} is not a valid permanent-tooth FDI number") from exc


def fdi_from_tooth_class_id(class_id: int) -> int:
    """Contiguous id in ``[0, 32)`` -> FDI number."""
    if not 0 <= class_id < NUM_TEETH:
        raise ValueError(f"tooth class id must be in [0, {NUM_TEETH}), got {class_id}")
    return FDI_NUMBERS[class_id]


def split_tooth_class_id(class_id: int) -> Tuple[int, int]:
    """Contiguous id in ``[0, 32)`` -> ``(quadrant, position)``, both 1-based.

    This is what makes the flat 32-way tooth head hierarchical for free: the
    quadrant and the position are recoverable from the class index.
    """
    if not 0 <= class_id < NUM_TEETH:
        raise ValueError(f"tooth class id must be in [0, {NUM_TEETH}), got {class_id}")
    quadrant, position = divmod(class_id, NUM_POSITIONS)
    return quadrant + 1, position + 1


def tooth_long_name(fdi: int) -> str:
    """``36 -> 'tooth 36 (lower-left first molar)'``."""
    quadrant, position = fdi_to_quadrant_position(fdi)
    side = QUADRANT_NAMES[quadrant - 1].split(" ", 1)[1]
    return f"tooth {fdi} ({side} {POSITION_NAMES[position - 1]})"


# --------------------------------------------------------------------------- #
# DENTEX category-id remapping
# --------------------------------------------------------------------------- #

def build_category_remap(
    categories: Sequence[dict],
    expected_names: Sequence[str],
) -> Dict[int, int]:
    """Map raw DENTEX category ids onto contiguous ids ordered by ``expected_names``.

    DENTEX's ``categories`` lists are keyed by ``id`` but only the ``name`` field
    is trustworthy (see the module docstring).  This resolves each entry by name
    and returns ``{raw_id: contiguous_id}``.

    Args:
        categories: the ``categories`` / ``categories_N`` list from a DENTEX json.
        expected_names: canonical names, in the order we want contiguous ids in.

    Raises:
        ValueError: if the json's names do not match ``expected_names`` as a set.
    """
    wanted = {str(name): i for i, name in enumerate(expected_names)}
    remap: Dict[int, int] = {}
    seen: List[str] = []
    for entry in categories:
        name = str(entry["name"])
        if name not in wanted:
            raise ValueError(
                f"unexpected DENTEX category name {name!r}; "
                f"expected one of {sorted(wanted)}"
            )
        remap[int(entry["id"])] = wanted[name]
        seen.append(name)
    missing = sorted(set(wanted) - set(seen))
    if missing:
        raise ValueError(f"DENTEX categories are missing {missing}")
    if len(seen) != len(set(seen)):
        raise ValueError(f"DENTEX categories contain duplicates: {seen}")
    return remap


def quadrant_remap(categories: Sequence[dict]) -> Dict[int, int]:
    """``{raw_id: quadrant_index}`` with ``quadrant_index`` in ``[0, 4)``."""
    return build_category_remap(categories, [str(q) for q in QUADRANTS])


def position_remap(categories: Sequence[dict]) -> Dict[int, int]:
    """``{raw_id: position_index}`` with ``position_index`` in ``[0, 8)``."""
    return build_category_remap(categories, [str(p) for p in POSITIONS])


def disease_remap(categories: Sequence[dict]) -> Dict[int, int]:
    """``{raw_id: disease_index}`` with ``disease_index`` in ``[0, 4)``."""
    return build_category_remap(categories, DISEASE_CLASS_NAMES)
