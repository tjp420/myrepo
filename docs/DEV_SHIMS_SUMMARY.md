**DEV Shims Summary (2025-12-12)

This document summarizes the development shims and lightweight fallbacks added to stabilize the FastAPI app in development mode (`AGI_DEV=1`) so smoke tests and local development run deterministically.

**Why**
- The production codebase depends on many optional, heavy subsystems (FAISS, Ollama, Redis, advanced ML libs). To enable fast local iteration and deterministic smoke testing without installing those heavy deps, small shims and safe fallbacks were added.

**Files added / changed (high level)**
- `agi_chatbot/performance/ultra_fast_mode.py`
  - Made `ultra_fast_enabled()` a callable (returns bool) to match call sites in `api_server.py`. Added `enable_ultra_fast()` / `disable_ultra_fast()` helpers.

- `agi_chatbot/metrics/runtime_metrics.py` (new)
  - Small in-memory shim providing `record_interaction`, `record_error`, `snapshot`, and `get_stats` APIs used by the server.

- `agi_chatbot/core/performance_optimizations.py`
  - Added `DummyEnhancedCache` fallback and ensured `EnhancedCache` resolves to a usable dummy in dev/test.

- `agi_chatbot/api_server.py`
  - `/cache/stats` now calls `get_enhanced_cache()` (lazy getter) and invokes `get_stats` via `_invoke_maybe_async` so sync or async cache shims are supported.
  - DEV_MODE fallbacks added: ensure lightweight `enhanced_cache`, `oracle` and `chatbot` stubs exist when heavy initialization is skipped.
  - Minor defensive changes to return degraded-but-valid responses where appropriate so smoke tests can continue.

**Behavior in DEV_MODE**
- Set `AGI_DEV=1` (or `true`) to enable `DEV_MODE` in `api_server.py`.
- The endpoints `/health`, `/cache/stats`, and `/chat` are safe to call and return predictable, JSON-serializable responses (may contain "dev_mode" placeholders or degraded data).

**How to run smoke tests locally (PowerShell)**
- Create / activate your virtualenv and install deps (if you haven't):

```powershell
# from repository root (Windows PowerShell)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:AGI_DEV = '1'
.\.venv\Scripts\python.exe tools\testclient_smoke.py
```

- To run the API in dev mode with uvicorn:

```powershell
$env:AGI_DEV = '1'
.\.venv\Scripts\python.exe -m uvicorn agi_chatbot.api_server:app --host 127.0.0.1 --port 8000
```

**Notes & Rationale**
- These shims are intentionally minimal and focused on enabling fast developer iteration and smoke testing. They are not production replacements for the missing services.
- Recommended approach: keep these shims clearly documented and gated by `AGI_DEV` (they were added this way) so production builds don't accidentally ship the shims.

**Recommended next steps**
- Run the full test suite (`pytest`) locally to detect any remaining regressions.
- Consolidate dev shims into a single module `agi_chatbot.dev_shims` to make their presence explicit.
- Add unit tests for the shimbed functions (`runtime_metrics`, `DummyEnhancedCache`, `ultra_fast_mode`) to prevent regressions.

**Contacts / Attribution**
- Changes were made during an iterative dev-run loop to stabilize imports and runtime behavior so the in-process TestClient could exercise endpoints deterministically.

