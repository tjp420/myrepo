# Contributing

Development setup (Windows PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) Upgrade packaging tools

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -U pip setuptools wheel
```

3) Choose a dev dependency set

- **Minimal (recommended for day-to-day development):** installs linting, testing and runtime deps.
  - Input: `requirements.dev.minimal.in`
  - Pinned output: `requirements.dev.minimal.txt`
  - Install (after activating venv):

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.dev.minimal.txt
```

- **ML toolchain (optional, heavy):** installs PyTorch, transformers and other ML packages for model development and experiments.
  - Input: `requirements.dev.ml.in`
  - Pinned output: `requirements.dev.ml.txt`
  - Install (note: large; may require CUDA/tooling):

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.dev.ml.txt
```

4) Install spaCy model (required by some features)

Option A — install from the pinned wheel (already pinned in `requirements*.txt` files):

```powershell
.\\.venv\\Scripts\\python.exe -m pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Option B — use spaCy's download command (recommended for convenience):

```powershell
.\\.venv\\Scripts\\python.exe -m spacy download en_core_web_sm
```

5) Run tests

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```

6) Run the API server (development)

```powershell
$env:AGI_DEV='1'
.\\.venv\\Scripts\\python.exe -m uvicorn agi_chatbot.api_server:app --host 127.0.0.1 --port 8000
```

Notes & recommendations
- Use `requirements.dev.minimal.txt` for quick iteration and CI jobs that don't need heavy ML toolchains.
- Use `requirements.dev.ml.txt` only when working on model training or experiments — it will install large native wheels and may require GPU drivers.
- We pin the spaCy model via a direct wheel URL so `pip-compile` can lock it; alternatively, contributors can use `python -m spacy download en_core_web_sm`.
- The repository contains many `.bak` files (~1.4k). Consider moving them to an `archive/` folder or a separate branch to reduce noise and speed tools.
- For CI reproducibility, prefer using the pinned `.txt` files and a clean venv; consider creating a `requirements.ci.in`/`requirements.ci.txt` if you need a bespoke CI set.
- Keep secrets out of the repo — use environment variables or a local `.env` file (do NOT commit secrets).

If you'd like, I can add sample one-line commands for creating a fresh test venv and performing a smoke install of the minimal dev pinned set on Windows.

Quick smoke test (one-liners)

These one-line commands create a temporary venv, upgrade packaging, install the minimal pinned dev set, and then show installed top-level packages. Use from PowerShell in the repo root.

```powershell
python -m venv .venv-smoke; .\.venv-smoke\Scripts\python.exe -m pip install -U pip setuptools wheel; .\.venv-smoke\Scripts\python.exe -m pip install -r requirements.dev.minimal.txt; .\.venv-smoke\Scripts\python.exe -m pip list --format=columns
```

To remove the temporary venv afterwards:

```powershell
Remove-Item -Recurse -Force .\.venv-smoke
```
