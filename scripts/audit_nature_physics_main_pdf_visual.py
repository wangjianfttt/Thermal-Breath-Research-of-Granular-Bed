#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
from pathlib import Path

from AppKit import (
    NSBitmapImageRep,
    NSCalibratedRGBColorSpace,
    NSColor,
    NSGraphicsContext,
    NSMakeRect,
    NSPDFImageRep,
    NSTIFFFileType,
)
from PIL import Image


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "thermal_ratcheting_nature_physics.pdf"
OUT_CSV = ROOT / "source_data" / "nature_physics_main_pdf_visual_qa.csv"
SCALE = 1.35
INK_THRESHOLD = 245


def render_page(pdf: NSPDFImageRep, page: int) -> Image.Image:
    pdf.setCurrentPage_(page)
    bounds = pdf.bounds()
    width = int(bounds.size.width * SCALE)
    height = int(bounds.size.height * SCALE)
    bitmap = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None,
        width,
        height,
        8,
        4,
        True,
        False,
        NSCalibratedRGBColorSpace,
        0,
        0,
    )
    context = NSGraphicsContext.graphicsContextWithBitmapImageRep_(bitmap)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(context)
    NSColor.whiteColor().set()
    NSMakeRect(0, 0, width, height)
    NSColor.whiteColor().setFill()
    from AppKit import NSRectFill

    NSRectFill(NSMakeRect(0, 0, width, height))
    pdf.drawInRect_(NSMakeRect(0, 0, width, height))
    NSGraphicsContext.restoreGraphicsState()
    data = bitmap.representationUsingType_properties_(NSTIFFFileType, None)
    return Image.open(io.BytesIO(bytes(data))).convert("RGB")


def page_metrics(image: Image.Image) -> tuple[float, float, float, str]:
    pixels = image.load()
    width, height = image.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if min(r, g, b) < INK_THRESHOLD:
                xs.append(x)
                ys.append(y)
    if not xs:
        return 0.0, 1.0, 1.0, "()"
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    area = ((right - left + 1) * (bottom - top + 1)) / (width * height)
    top_blank = top / height
    bottom_blank = (height - bottom - 1) / height
    return area, top_blank, bottom_blank, f"({left}, {top}, {right}, {bottom})"


def main() -> None:
    pdf = NSPDFImageRep.imageRepWithContentsOfFile_(str(PDF))
    if pdf is None:
        raise SystemExit(f"Could not open {PDF}")
    rows = []
    for page in range(pdf.pageCount()):
        image = render_page(pdf, page)
        area, top_blank, bottom_blank, bbox = page_metrics(image)
        rows.append(
            {
                "rendered_page": f"main_page_{page + 1:02d}.png",
                "content_area_fraction": f"{area:.6f}",
                "top_blank_fraction": f"{top_blank:.6f}",
                "bottom_blank_fraction": f"{bottom_blank:.6f}",
                "bbox": bbox,
            }
        )
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(rows)} rendered pages.")


if __name__ == "__main__":
    main()
