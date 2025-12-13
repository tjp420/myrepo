"""Run pylint and write JSON output as bytes to pylint_errors.json."""

import subprocess
import sys
from pathlib import Path


def main():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pylint",
            "agi_chatbot",
            "--disable=R,C",
            "--output-format=json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = proc.stdout if proc.stdout else proc.stderr
    Path("pylint_errors.json").write_bytes(out or b"[]")


if __name__ == "__main__":
    main()
