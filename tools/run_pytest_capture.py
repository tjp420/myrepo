import subprocess
import sys
from pathlib import Path

out = Path("pytest_oracle_routing.txt")
proc = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/unit/test_oracle_routing.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
out.write_text(proc.stdout, encoding="utf-8")
print(f"Wrote pytest output to {out} (exit {proc.returncode})")
sys.exit(proc.returncode)
