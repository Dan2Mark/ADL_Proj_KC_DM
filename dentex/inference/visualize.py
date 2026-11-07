"""Draw findings on a panoramic x-ray.

Two box layers, because they answer two different questions:

* **teeth** -- thin cool-coloured boxes labelled with the FDI number, so you can
  see *which tooth is which*;
* **problems** -- thick warm-coloured boxes labelled ``"47 Deep Caries 0.81"``,
  so you can see *what is wrong and where*.

Panoramic x-rays are around 2900x1500 px, so stroke widths and font sizes are
derived from the image diagonal instead of being hard-coded.  Only Pillow and
NumPy are needed -- no OpenCV, no detectron2 ``Visualizer`` -- which keeps this
usable in a plain notebook and in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import fdi
from .report import ProblemFinding, StudyReport, ToothFinding

RGB = Tuple[int, int, int]

#: One colour per diagnosis, kept consistent across every figure in the project.
DISEASE_COLORS: dict[str, RGB] = {
    "Impacted": (255, 176, 46),         # amber
    "Caries": (255, 86, 86),            # red
    "Periapical Lesion": (205, 99, 255),  # violet
    "Deep Caries": (214, 25, 61),       # deep red
}

TOOTH_COLOR: RGB = (64, 214, 190)       # teal
TOOTH_COLOR_FLAGGED: RGB = (255, 255, 255)
TEXT_COLOR: RGB = (12, 14, 18)

_FONT_CANDIDATES = (
    "DejaVuSans-Bold.ttf",   # Linux / Colab
    "DejaVuSans.ttf",
    "arialbd.ttf",           # Windows
    "arial.ttf",
    "segoeui.ttf",
    "Helvetica.ttc",         # macOS
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _scale(width: int, height: int) -> float:
    """Scale factor so that strokes and text read the same on any x-ray size."""
    diagonal = float(np.hypot(width, height))
    return max(diagonal / 3200.0, 0.35)


def _as_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def load_xray(path: str | Path) -> Image.Image:
    """Open an x-ray as RGB.  DENTEX pngs are already RGB, but be safe."""
    return _as_rgb(Image.open(path))


def polygons_to_mask(
    polygons: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterise flat ``[x1, y1, x2, y2, ...]`` polygon parts into a bool mask.

    Used to visualise DENTEX ground-truth polygons without pulling in
    ``pycocotools``; predicted masks already arrive as rasters.
    """
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for part in polygons:
        if len(part) < 6:
            continue
        points = [(float(part[i]), float(part[i + 1])) for i in range(0, len(part) - 1, 2)]
        drawer.polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    color: RGB,
    width: int,
) -> None:
    x1, y1, x2, y2 = (float(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=max(width, 1))


def _overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _draw_label(
    draw: ImageDraw.ImageDraw,
    anchor: Tuple[float, float],
    text: str,
    color: RGB,
    font: ImageFont.ImageFont,
    pad: int,
    above: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
    occupied: Optional[List[Tuple[float, float, float, float]]] = None,
) -> None:
    """Draw ``text`` in a filled chip, kept inside the image and off other chips.

    Panoramic x-rays routinely have several findings on neighbouring teeth, so
    labels are nudged along the drawing direction until they stop colliding with
    chips already placed (tracked in ``occupied``).
    """
    x, y = anchor
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    box_w, box_h = tw + 2 * pad, th + 2 * pad

    y0 = y - box_h if above else y
    x0 = x

    def clamp(px: float, py: float) -> Tuple[float, float]:
        if image_size is None:
            return px, py
        iw, ih = image_size
        return (
            min(max(0.0, px), max(0.0, iw - box_w)),
            min(max(0.0, py), max(0.0, ih - box_h)),
        )

    x0, y0 = clamp(x0, y0)
    if occupied is not None:
        step = box_h + max(pad // 2, 1)
        direction = -1 if above else 1
        for attempt in range(12):
            rect = (x0, y0, x0 + box_w, y0 + box_h)
            if not any(_overlaps(rect, other) for other in occupied):
                break
            _, y0 = clamp(x0, y0 + direction * step)
            if attempt == 5:  # ran out of room vertically -- try the other side
                direction *= -1
        occupied.append((x0, y0, x0 + box_w, y0 + box_h))

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=color)
    draw.text((x0 + pad - left, y0 + pad - top), text, fill=TEXT_COLOR, font=font)


def _overlay_masks(
    image: Image.Image,
    masks: Iterable[np.ndarray],
    colors: Iterable[RGB],
    alpha: float,
) -> Image.Image:
    """Blend binary masks into the image at the given opacity."""
    base = np.asarray(image, dtype=np.float32)
    for mask, color in zip(masks, colors):
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.shape[:2] != base.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape[:2]} does not match image {base.shape[:2]}"
            )
        selection = mask.astype(bool)
        if not selection.any():
            continue
        tint = np.asarray(color, dtype=np.float32)
        base[selection] = (1.0 - alpha) * base[selection] + alpha * tint
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def draw_report(
    image: str | Path | Image.Image,
    report: StudyReport,
    draw_teeth: bool = True,
    draw_problems: bool = True,
    show_masks: bool = True,
    mask_alpha: float = 0.35,
    tooth_label: str = "fdi",
    min_tooth_score: float = 0.0,
    min_problem_score: float = 0.0,
    legend: bool = True,
    title: Optional[str] = None,
) -> Image.Image:
    """Render a :class:`~dentex.inference.report.StudyReport` onto its x-ray.

    Args:
        image: path to the x-ray, or an already-open ``PIL.Image``.
        report: findings to draw.
        draw_teeth: draw the tooth layer (thin teal boxes with FDI numbers).
        draw_problems: draw the pathology layer (thick coloured boxes).
        show_masks: blend instance masks under the boxes, where available.
        mask_alpha: opacity of the mask overlay in ``[0, 1]``.
        tooth_label: ``"fdi"`` for just the number, ``"full"`` to add the score,
            ``"none"`` to label nothing.
        min_tooth_score: hide teeth scoring below this.
        min_problem_score: hide problems scoring below this.
        legend: draw a legend listing the diagnosis colours that appear.
        title: optional caption drawn in the top-left corner.

    Returns:
        A new ``PIL.Image``; the input is never modified.
    """
    canvas = load_xray(image) if not isinstance(image, Image.Image) else _as_rgb(image.copy())
    width, height = canvas.size
    scale = _scale(width, height)

    teeth: List[ToothFinding] = (
        [t for t in report.teeth if t.score >= min_tooth_score] if draw_teeth else []
    )
    problems: List[ProblemFinding] = (
        [p for p in report.problems if p.score >= min_problem_score] if draw_problems else []
    )

    if show_masks:
        masks = [t.mask for t in teeth] + [p.mask for p in problems]
        colors = [TOOTH_COLOR] * len(teeth) + [
            DISEASE_COLORS[p.disease] for p in problems
        ]
        if any(m is not None for m in masks):
            canvas = _overlay_masks(canvas, masks, colors, mask_alpha)

    draw = ImageDraw.Draw(canvas)
    tooth_font = _load_font(max(int(26 * scale), 10))
    problem_font = _load_font(max(int(34 * scale), 12))
    legend_font = _load_font(max(int(30 * scale), 11))
    pad = max(int(6 * scale), 2)

    flagged = {p.tooth_id for p in problems if p.tooth_id is not None}
    placed_above: List[Tuple[float, float, float, float]] = []
    placed_below: List[Tuple[float, float, float, float]] = []

    # Tooth layer first, so pathology boxes sit on top of it.
    for tooth in sorted(teeth, key=lambda t: t.bbox[0]):
        color = TOOTH_COLOR_FLAGGED if tooth.tooth_id in flagged else TOOTH_COLOR
        _draw_box(draw, tooth.bbox, color, int(max(2 * scale, 1)))
        if tooth_label != "none":
            text = str(tooth.fdi)
            if tooth_label == "full":
                text = f"{tooth.fdi} {tooth.score:.2f}"
            _draw_label(
                draw,
                (tooth.bbox[0], tooth.bbox[1]),
                text,
                color,
                tooth_font,
                pad,
                above=True,
                image_size=canvas.size,
                occupied=placed_above,
            )

    # Pathology layer: thicker stroke, label below the box so it does not
    # collide with the tooth number drawn above it.
    for problem in sorted(problems, key=lambda p: (p.bbox[0], -p.score)):
        color = DISEASE_COLORS[problem.disease]
        _draw_box(draw, problem.bbox, color, int(max(5 * scale, 2)))
        tooth_part = str(problem.fdi) if problem.fdi is not None else "?"
        text = f"{tooth_part} - {problem.disease} {problem.score:.2f}"
        _draw_label(
            draw,
            (problem.bbox[0], problem.bbox[3]),
            text,
            color,
            problem_font,
            pad,
            above=False,
            image_size=canvas.size,
            occupied=placed_below,
        )

    if legend and problems:
        present = []
        for name in fdi.DISEASE_CLASS_NAMES:
            if any(p.disease == name for p in problems):
                present.append(name)
        _draw_legend(draw, canvas.size, present, legend_font, pad, scale, bool(teeth))

    if title:
        _draw_label(
            draw,
            (pad * 2, pad * 2),
            title,
            (235, 238, 245),
            legend_font,
            pad,
            above=False,
            image_size=canvas.size,
        )

    return canvas


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    size: Tuple[int, int],
    diseases: Sequence[str],
    font: ImageFont.ImageFont,
    pad: int,
    scale: float,
    with_teeth: bool,
) -> None:
    width, height = size
    entries: List[Tuple[str, RGB]] = [(name, DISEASE_COLORS[name]) for name in diseases]
    if with_teeth:
        entries.append(("tooth (FDI)", TOOTH_COLOR))
    if not entries:
        return

    swatch = max(int(22 * scale), 8)
    gap = max(int(10 * scale), 4)
    line_h = swatch + gap
    text_w = max(
        draw.textbbox((0, 0), name, font=font)[2] for name, _ in entries
    )
    box_w = swatch + gap + text_w + 4 * pad
    box_h = line_h * len(entries) + 2 * pad

    x0 = width - box_w - 2 * pad
    y0 = height - box_h - 2 * pad
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(18, 20, 26))

    y = y0 + pad
    for name, color in entries:
        draw.rectangle([x0 + 2 * pad, y, x0 + 2 * pad + swatch, y + swatch], fill=color)
        draw.text(
            (x0 + 2 * pad + swatch + gap, y),
            name,
            fill=(235, 238, 245),
            font=font,
        )
        y += line_h


# --------------------------------------------------------------------------- #
# Figures for the write-up
# --------------------------------------------------------------------------- #

def side_by_side(
    left: Image.Image,
    right: Image.Image,
    gap: int = 16,
    background: RGB = (18, 20, 26),
) -> Image.Image:
    """Stack two renders vertically -- e.g. ground truth above predictions."""
    width = max(left.width, right.width)
    height = left.height + right.height + gap
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(left, ((width - left.width) // 2, 0))
    canvas.paste(right, ((width - right.width) // 2, left.height + gap))
    return canvas


def contact_sheet(
    images: Sequence[Image.Image],
    columns: int = 2,
    thumb_width: int = 900,
    gap: int = 12,
    background: RGB = (18, 20, 26),
) -> Image.Image:
    """Tile several renders into one figure."""
    if not images:
        raise ValueError("contact_sheet needs at least one image")
    thumbs = []
    for img in images:
        ratio = thumb_width / img.width
        thumbs.append(img.resize((thumb_width, max(1, int(img.height * ratio)))))

    rows = (len(thumbs) + columns - 1) // columns
    row_heights = [
        max(t.height for t in thumbs[r * columns : (r + 1) * columns])
        for r in range(rows)
    ]
    canvas = Image.new(
        "RGB",
        (columns * thumb_width + (columns + 1) * gap, sum(row_heights) + (rows + 1) * gap),
        background,
    )
    y = gap
    for r in range(rows):
        x = gap
        for thumb in thumbs[r * columns : (r + 1) * columns]:
            canvas.paste(thumb, (x, y))
            x += thumb_width + gap
        y += row_heights[r] + gap
    return canvas


__all__ = [
    "DISEASE_COLORS",
    "TOOTH_COLOR",
    "contact_sheet",
    "draw_report",
    "load_xray",
    "side_by_side",
]
