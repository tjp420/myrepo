# Histogram Thumbnails — Sample Large Dataset

This folder contains the artifacts produced during an analysis session that generated three alternative normalized histograms from `sample_large_dataset.csv` and assembled them into a self-contained HTML wrapper.

Files
- `histogram_200_normalized.png` — 200-bin normalized histogram PNG
- `histogram_log_signedlog_normalized.png` — signed-log histogram PNG
- `kde_approx_normalized.png` — KDE approximation PNG
- `histogram_thumbnails_wrapper.html` — self-contained HTML with embedded thumbnails (data URIs)
- `histogram_thumbnails_wrapper.png` — headless screenshot of the wrapper (generated via Playwright)
- `histogram_thumbnails_wrapper.png.b64` — base64-encoded PNG (same image as `.png`)
- `render_wrapper_screenshot.py` — Playwright script used to render the wrapper to PNG
- `view_histogram_thumbnails.html` — convenience HTML that embeds the PNG data URI for quick viewing

Purpose
- Provide a compact, self-contained way to review the three histogram visualizations without external dependencies.
- Enable reproducible headless rendering (Playwright) to capture consistent screenshots.

How to restore the PNG locally
PowerShell:
```powershell
$in = 'histogram_thumbnails_wrapper.png.b64'
[System.IO.File]::WriteAllBytes('histogram_thumbnails_wrapper_restored.png', [Convert]::FromBase64String((Get-Content $in -Raw)))
```

Python:
```python
import base64
open('histogram_thumbnails_wrapper_restored.png','wb').write(base64.b64decode(open('histogram_thumbnails_wrapper.png.b64','rb').read()))
```

Notes
- The Playwright Chromium runtime used to generate `histogram_thumbnails_wrapper.png` was downloaded via `python -m playwright install chromium` on the environment where this work was run.
- If you want these artifacts committed and pushed to a remote, confirm and I'll push the current branch.
