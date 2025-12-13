#!/usr/bin/env python3
import base64
import io
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
EMBEDDED = OUT_DIR / "histograms_normalized_embedded.html"
PREVIEW = OUT_DIR / "histograms_normalized_preview.html"
SIGNED_LOG_PNG = OUT_DIR / "histogram_log_signedlog_normalized.png"


def check_embedded_html(path):
    txt = path.read_text(encoding="utf-8")
    img_tags = len(re.findall(r"<img\b", txt))
    data_uris = len(re.findall(r"data:image/png;base64,", txt))
    return img_tags, data_uris


def check_preview_html(path):
    txt = path.read_text(encoding="utf-8")
    srcs = re.findall(r'src="([^"]+)"', txt)
    return srcs


def make_thumbnail_base64(png_path, width=400):
    arr = mpimg.imread(str(png_path))
    # arr shape: (H,W,3/4)
    h, w = arr.shape[0], arr.shape[1]
    if w > width:
        factor = w / width
        new_h = max(1, int(h / factor))
        # simple nearest-neighbor downsample
        ys = (np.linspace(0, h - 1, new_h)).astype(int)
        xs = (np.linspace(0, w - 1, width)).astype(int)
        arr = arr[ys[:, None], xs]

    buf = io.BytesIO()
    # save with matplotlib to buffer
    plt.imsave(buf, arr, format="png")
    b = buf.getvalue()
    return base64.b64encode(b).decode("ascii")


def main():
    out = []
    out.append(f"Embedded HTML: {EMBEDDED}")
    if EMBEDDED.exists():
        tags, uris = check_embedded_html(EMBEDDED)
        out.append(f"  img tags: {tags}, data URI png count: {uris}")
    else:
        out.append("  MISSING")

    out.append(f"Preview HTML: {PREVIEW}")
    if PREVIEW.exists():
        srcs = check_preview_html(PREVIEW)
        out.append(f"  img src count: {len(srcs)}")
        missing = [s for s in srcs if not (OUT_DIR / s).exists()]
        out.append(f"  referenced files missing: {len(missing)}")
        if missing:
            out.append(f"    {missing[:10]}")
    else:
        out.append("  MISSING")

    out.append(f"Signed-log PNG: {SIGNED_LOG_PNG} (exists: {SIGNED_LOG_PNG.exists()})")

    print("\n".join(out))

    if SIGNED_LOG_PNG.exists():
        try:
            b64 = make_thumbnail_base64(SIGNED_LOG_PNG, width=400)
            print("---BASE64_THUMBNAIL_START---")
            print(b64)
            print("---BASE64_THUMBNAIL_END---")
        except Exception as e:
            print("Thumbnail generation failed:", e)


if __name__ == "__main__":
    main()
