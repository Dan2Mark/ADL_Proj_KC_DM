"""Fetch the DENTEX release from the Hugging Face Hub.

The Hub copy (``ibrahimhamamci/DENTEX``) ships three archives::

    DENTEX/training_data.zip      10.9 GB   693 + 634 + 705 labelled + 1571 unlabelled x-rays
    DENTEX/validation_data.zip     149 MB   the 50 official validation x-rays
    DENTEX/test_data.zip           765 MB   the 250 official test x-rays (no labels here)
    DENTEX/validation_triple.json   64 KB   ground truth for the 50 validation x-rays

Downloading all of ``training_data.zip`` is wasteful for this project: more than
half of it is the ``unlabelled/`` pre-training split, which the supervised
pipeline never touches.  :func:`fetch_training_data` therefore defaults to
*streaming* mode -- it reads the zip's central directory over HTTP range
requests and pulls out only the members belonging to the subsets you asked for.
Pass ``mode="full"`` to download the whole archive instead (useful if you want
the unlabelled split for self-supervised pre-training, or if range requests are
blocked on your network).

Layout produced under ``root``::

    root/
      training_data/
        quadrant/{train_quadrant.json,xrays/*.png}
        quadrant_enumeration/{train_quadrant_enumeration.json,xrays/*.png}
        quadrant-enumeration-disease/{train_quadrant_enumeration_disease.json,xrays/*.png}
      validation_data/quadrant_enumeration_disease/xrays/*.png
      validation_triple.json
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO_ID = "ibrahimhamamci/DENTEX"
REPO_TYPE = "dataset"

TRAINING_ZIP = "DENTEX/training_data.zip"
VALIDATION_ZIP = "DENTEX/validation_data.zip"
TEST_ZIP = "DENTEX/test_data.zip"
VALIDATION_TRIPLE = "DENTEX/validation_triple.json"

#: Directory names of the labelled subsets *inside* ``training_data.zip``.
#: Note the inconsistent separators upstream -- ``quadrant_enumeration`` uses
#: underscores while ``quadrant-enumeration-disease`` uses hyphens.
SUBSET_DIRS = {
    "quadrant": "training_data/quadrant",
    "enumeration": "training_data/quadrant_enumeration",
    "disease": "training_data/quadrant-enumeration-disease",
    "unlabelled": "training_data/unlabelled",
}

#: Subsets the supervised pipeline needs: teeth come from ``enumeration``,
#: pathology from ``disease``, and ``quadrant`` is the optional warm-up stage.
DEFAULT_SUBSETS: Sequence[str] = ("quadrant", "enumeration", "disease")

logger = logging.getLogger(__name__)


def _hf_filesystem():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def _remote_zip_path(filename: str) -> str:
    return f"datasets/{REPO_ID}/{filename}"


def _members_for(subsets: Iterable[str]) -> List[str]:
    unknown = sorted(set(subsets) - set(SUBSET_DIRS))
    if unknown:
        raise ValueError(f"unknown subset(s) {unknown}; choose from {sorted(SUBSET_DIRS)}")
    return [SUBSET_DIRS[s] + "/" for s in subsets]


def _extract_members(
    zf: zipfile.ZipFile,
    dest: Path,
    prefixes: Sequence[str],
    overwrite: bool,
) -> int:
    written = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        if prefixes and not any(info.filename.startswith(p) for p in prefixes):
            continue
        target = dest / info.filename
        if target.exists() and not overwrite and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        written += 1
        if written % 100 == 0:
            logger.info("extracted %d files...", written)
    return written


def fetch_training_data(
    root: str | Path,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    mode: str = "stream",
    overwrite: bool = False,
) -> Path:
    """Materialise the requested training subsets under ``root``.

    Args:
        root: destination directory; ``training_data/`` is created inside it.
        subsets: any of ``quadrant``, ``enumeration``, ``disease``, ``unlabelled``.
        mode: ``"stream"`` pulls only the requested members straight out of the
            remote zip (no 10.9 GB download); ``"full"`` downloads the archive
            to the Hub cache first and then unzips locally.
        overwrite: re-extract files that already exist at the right size.

    Returns:
        The ``root`` path.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefixes = _members_for(subsets)

    if mode == "stream":
        logger.info("streaming %s from the Hub (subsets: %s)", TRAINING_ZIP, list(subsets))
        fs = _hf_filesystem()
        handle = fs.open(_remote_zip_path(TRAINING_ZIP), "rb")
        with zipfile.ZipFile(handle) as zf:
            n = _extract_members(zf, root, prefixes, overwrite)
    elif mode == "full":
        from huggingface_hub import hf_hub_download

        logger.info("downloading the whole %s (~10.9 GB)", TRAINING_ZIP)
        local = hf_hub_download(REPO_ID, TRAINING_ZIP, repo_type=REPO_TYPE)
        with zipfile.ZipFile(local) as zf:
            n = _extract_members(zf, root, prefixes, overwrite)
    else:
        raise ValueError(f"mode must be 'stream' or 'full', got {mode!r}")

    logger.info("wrote %d new file(s) under %s", n, root)
    return root


def fetch_validation_data(root: str | Path, overwrite: bool = False) -> Path:
    """Download the 50-image official validation split and its ground truth.

    DENTEX released the validation labels after the challenge closed, as
    ``validation_triple.json``.  The project uses this split as an untouched
    hold-out test set, so that the 705 fully labelled training x-rays can be
    split into train/val without leaking.
    """
    from huggingface_hub import hf_hub_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(REPO_ID, VALIDATION_ZIP, repo_type=REPO_TYPE)
    with zipfile.ZipFile(zip_path) as zf:
        _extract_members(zf, root, prefixes=(), overwrite=overwrite)

    gt = hf_hub_download(REPO_ID, VALIDATION_TRIPLE, repo_type=REPO_TYPE)
    shutil.copyfile(gt, root / "validation_triple.json")
    logger.info("validation split ready under %s", root)
    return root


def fetch_test_data(root: str | Path, overwrite: bool = False) -> Path:
    """Download the 250-image official test split (images only, no labels)."""
    from huggingface_hub import hf_hub_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(REPO_ID, TEST_ZIP, repo_type=REPO_TYPE)
    with zipfile.ZipFile(zip_path) as zf:
        _extract_members(zf, root, prefixes=(), overwrite=overwrite)
    return root


def locate(root: str | Path, subset: str) -> tuple[Path, Path]:
    """Return ``(annotation_json, images_dir)`` for a materialised subset.

    Raises:
        FileNotFoundError: if the subset has not been downloaded yet.
    """
    root = Path(root)
    if subset not in SUBSET_DIRS:
        raise ValueError(f"unknown subset {subset!r}; choose from {sorted(SUBSET_DIRS)}")
    base = root / SUBSET_DIRS[subset]
    images = base / "xrays"
    if not images.is_dir():
        raise FileNotFoundError(
            f"{images} is missing -- run fetch_training_data(root, subsets=['{subset}']) first"
        )
    jsons = sorted(base.glob("*.json"))
    if subset == "unlabelled":
        return (Path(), images)
    if not jsons:
        raise FileNotFoundError(f"no annotation json found in {base}")
    return (jsons[0], images)


def find_validation(root: str | Path) -> tuple[Path, Path]:
    """Return ``(validation_triple.json, images_dir)`` for the hold-out split."""
    root = Path(root)
    gt = root / "validation_triple.json"
    if not gt.is_file():
        raise FileNotFoundError(f"{gt} is missing -- run fetch_validation_data(root) first")
    candidates = [p for p in root.rglob("xrays") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no xrays/ directory found under {root}")
    # Prefer the deepest match, which is the quadrant_enumeration_disease folder.
    images = max(candidates, key=lambda p: len(p.parts))
    return gt, images


__all__ = [
    "DEFAULT_SUBSETS",
    "SUBSET_DIRS",
    "fetch_training_data",
    "fetch_validation_data",
    "fetch_test_data",
    "locate",
    "find_validation",
]
