"""
Simple test harness that runs plain `test_*` functions in modules under `tests/`.

This avoids introducing a dependency on `pytest`. It imports each test
module, finds callables whose name starts with `test_`, and executes
them, reporting failures and a non-zero exit code when any test fails.

Usage:
    python tools\run_tests.py
"""

import importlib.util
import os
import sys
import traceback


def discover_test_files(tests_dir="tests"):
    for entry in sorted(os.listdir(tests_dir)):
        if entry.startswith("test_") and entry.endswith(".py"):
            yield os.path.join(tests_dir, entry)


def load_module_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_tests():
    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests")
    tests_dir = os.path.abspath(tests_dir)
    if not os.path.isdir(tests_dir):
        print(f"No tests directory at {tests_dir}")
        return 0

    # Make sure repository root is on sys.path so `agi_chatbot` imports work
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Install a simple stderr filter to suppress repeated noisy warnings
    # about optional components. This keeps the test output cleaner while
    # still showing the first occurrence of each important message.
    import re

    class FilteredStderr:
        def __init__(self, wrapped, patterns=None):
            self._w = wrapped
            self._seen = set()
            self._patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or [])]

        def write(self, s):
            if not s:
                return
            # If line matches a pattern we've already seen, suppress it.
            for p in self._patterns:
                if p.search(s):
                    key = p.pattern
                    if key in self._seen:
                        return
                    self._seen.add(key)
                    break
            try:
                self._w.write(s)
            except Exception:
                pass

        def flush(self):
            try:
                self._w.flush()
            except Exception:
                pass

    noisy_patterns = [
        r"divine optimizer not available",
        r"performance patch not found",
        r"quality scorer/templates not available",
        r"performance enhancement system not available",
        r"not available",
    ]
    sys.stderr = FilteredStderr(sys.stderr, patterns=noisy_patterns)

    total = 0
    failed = 0
    failures = []

    for path in discover_test_files(tests_dir):
        print(f"Discovering {path}")
        try:
            mod = load_module_from_path(path)
        except Exception:
            print(f"Failed to import {path}")
            traceback.print_exc()
            failed += 1
            continue

        for name in dir(mod):
            if not name.startswith("test_"):
                continue
            obj = getattr(mod, name)
            if callable(obj):
                total += 1
                try:
                    print(f"RUN {name}...", end=" ")
                    obj()
                    print("OK")
                except AssertionError:
                    print("FAIL")
                    failed += 1
                    failures.append(
                        (path, name, "AssertionError", traceback.format_exc())
                    )
                except Exception as e:
                    print("ERROR")
                    failed += 1
                    failures.append(
                        (path, name, type(e).__name__, traceback.format_exc())
                    )

    print()
    print(f"Ran {total} tests: {total - failed} passed, {failed} failed")
    if failed:
        print("\nFailures detail:")
        for path, name, errtype, tb in failures:
            print("---")
            print(f"{path}::{name} -> {errtype}")
            print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    rc = run_tests()
    sys.exit(rc)
