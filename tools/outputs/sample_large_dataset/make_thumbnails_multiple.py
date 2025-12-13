from pathlib import Path

from PIL import Image

SRC = Path(
    r"E:\Ai\AGI Chatbot\tools\outputs\sample_large_dataset\histogram_thumbnails_wrapper_restored.png"
)
OUT_DIR = SRC.parent
SIZES = [(800, 600), (640, 480), (320, 240)]


def make_thumbnail(size):
    out = OUT_DIR / f"histogram_thumbnails_wrapper_{size[0]}x{size[1]}.png"
    im = Image.open(SRC)
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
    im.thumbnail(size, resample)
    im.save(out, optimize=True)
    return out


def main():
    results = []
    for s in SIZES:
        out = make_thumbnail(s)
        results.append(out)
        print("Wrote:", out)
    for p in results:
        info = p.stat()
        print(p.name, info.st_size)


if __name__ == "__main__":
    main()
