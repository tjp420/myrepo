## [v1.0.1] - 2025-12-15
- Added aggregation and plotting artifacts under `runtime_artifacts/`:
  - `runtime_artifacts/aggregated_results.json`
  - `runtime_artifacts/plots/*`
- Added helper tooling: `tools/aggregate_telemetry.py`, `tools/plot_aggregated_metrics.py`.
- Recreated and attached `release_ocr_results_20251215_140416.zip` to the `v1.0.1` GitHub releases.

(See `CHANGELOG.md` for full history.)

## Usage examples

1) Generate aggregated results from runtime telemetry JSON files:

```powershell
python tools/aggregate_telemetry.py \
  --input runtime_artifacts/load_telemetry_*.json \
  --output runtime_artifacts/aggregated_results.json
```

2) Create plots from the aggregated results:

```powershell
python tools/plot_aggregated_metrics.py \
  runtime_artifacts/aggregated_results.json \
  --out runtime_artifacts/plots
```

3) Download the release ZIP (bash):

```bash
curl -L -o release_ocr_results_20251215_140416.zip \
  https://github.com/tjp420/myrepo/releases/download/v1.0.1/release_ocr_results_20251215_140416.zip
unzip release_ocr_results_20251215_140416.zip -d release_artifacts
```

4) Verify checksum (Windows PowerShell example):

```powershell
Get-FileHash runtime_artifacts\releases\release_ocr_results_20251215_140416.zip -Algorithm SHA256 | Format-List
# Expected: 83B02F3411086D4C60E71CC68BFA838467B1814305FAD560744565C5BAE53145
```

---

SHA-256 (release ZIP): 83B02F3411086D4C60E71CC68BFA838467B1814305FAD560744565C5BAE53145
