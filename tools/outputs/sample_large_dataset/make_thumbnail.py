from pathlib import Path

from PIL import Image

SRC = Path(
    r"E:\Ai\AGI Chatbot\tools\outputs\sample_large_dataset\histogram_thumbnails_wrapper_restored.png"
)
OUT = SRC.with_name("histogram_thumbnails_wrapper_thumbnail.png")

im = Image.open(SRC)
try:
    resample = Image.Resampling.LANCZOS
except Exception:
    resample = getattr(Image, "LANCZOS", Image.BICUBIC)

im.thumbnail((1200, 600), resample)
im.save(OUT, optimize=True)
print("Wrote:", OUT)
