#!/usr/bin/env python3
"""
Generate base64 thumbnails for the three PNGs and print full HTML <img> tags.
Outputs are printed to stdout separated by markers for easy copy/paste.
"""
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
FILES = [
    (OUT_DIR / "histogram_200_normalized.png", "Histogram (normalized, 200 bins)"),
    (
        OUT_DIR / "histogram_log_signedlog_normalized.png",
        "Signed-log histogram (normalized)",
    ),
    (OUT_DIR / "kde_approx_normalized.png", "KDE (normalized)"),
]


def make_thumb_b64(path: Path, width=400):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    if w > width:
        new_h = max(1, int(h * (width / w)))
        img = img.resize((width, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    for p, caption in FILES:
        print(f"---IMG_TAG_START: {p.name}---")
        if not p.exists():
            print(f"<!-- MISSING: {p} -->")
            print(f"---IMG_TAG_END: {p.name}---\n")
            continue
        b64 = make_thumb_b64(p, width=600)
        tag = f'<img src="data:image/png;base64,{b64}" alt="{caption}" style="max-width:32%;height:auto;border:1px solid #ccc;padding:4px;background:#fff">'
        print(tag)
        print(f"---IMG_TAG_END: {p.name}---\n")


if __name__ == "__main__":
    main()
