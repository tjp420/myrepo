import base64
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

FILES = [
    "histogram_200_normalized.png",
    "histogram_log_signedlog_normalized.png",
    "kde_approx_normalized.png",
]

out_html = OUT_DIR / "histogram_thumbnails_wrapper.html"


def img_data_uri(p: Path) -> str:
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main():
    imgs = []
    for fn in FILES:
        p = OUT_DIR / fn
        if not p.exists():
            raise FileNotFoundError(f"Expected image not found: {p}")
        imgs.append(img_data_uri(p))

    html_parts = [
        "<!doctype html>",
        '<html lang="en">',
        '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Histogram Thumbnails</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;background:#f6f6f6;padding:20px;}",
        ".wrap{display:flex;gap:12px;align-items:flex-start;}",
        ".thumb{border:1px solid #ccc;padding:6px;background:#fff;max-width:200px;width:200px;height:150px;object-fit:contain;}",
        "</style></head>",
        "<body>",
        "<h3>Histogram Thumbnails (normalized)</h3>",
        '<div class="wrap">',
    ]

    for src in imgs:
        html_parts.append(f'<img class="thumb" src="{src}" width="200" height="150">')

    html_parts.extend(["</div>", "</body>", "</html>"])

    out_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
