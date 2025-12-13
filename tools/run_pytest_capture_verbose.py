import os
import sys
import traceback

out_path = os.path.join(os.getcwd(), "pytest_oracle_routing.txt")

with open(out_path, "w", encoding="utf-8") as f:
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = f
        sys.stderr = f
        try:
            import pytest
        except Exception as exc:
            f.write("ERROR: failed to import pytest: %r\n" % exc)
            traceback.print_exc(file=f)
            f.flush()
            sys.exit(1)

        try:
            rc = pytest.main(["-q", "tests/unit/test_oracle_routing.py"])
            f.write("\n=== PYTEST EXIT CODE: %s ===\n" % rc)
            f.flush()
        except SystemExit as se:
            f.write("\n=== PYTEST raised SystemExit: %s ===\n" % se.code)
            f.flush()
            rc = se.code if isinstance(se.code, int) else 1
        except Exception:
            f.write("\n=== PYTEST raised exception ===\n")
            traceback.print_exc(file=f)
            f.flush()
            rc = 2
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

print("Wrote pytest output to", out_path)
sys.exit(rc)
