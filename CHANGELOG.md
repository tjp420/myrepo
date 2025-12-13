## Unreleased

- PR: create changelog PR to review upcoming entries.


## v0.1.1 — 2025-12-13

- Release v0.1.1 — Add pinned minimal dev/test requirements and CI improvements; added `pyproject.toml` so `-e .` installs in CI; fixed test-collection and exported `get_enhanced_framework` stub so tests can monkeypatch it; CI: green.

- Add `agi_chatbot/optional_imports.py` helper to centralize optional imports and dedupe warnings.

- Wire defensive optional imports in `agi_chatbot/core/enhanced_answer.py` and `agi_chatbot/api_server.py` so tests and CI remain quiet when optional components are absent.

- Small flake8-fixes to newly added helper and warning message text.

- Updated tests and harness to reduce noisy warnings coming from optional subsystems.

Release URL: https://github.com/tjp420/agi-chatbot/releases/tag/v0.1.1
