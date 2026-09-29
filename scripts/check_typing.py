"""Require each checker to reject each marked invalid physical expression."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/typing/negative/invalid.py"
UV = shutil.which("uv")
COMMANDS = (
    ("mypy", "--strict"),
    ("pyright",),
    ("pyrefly", "check"),
    ("ty", "check", "--output-format", "concise"),
)


def main() -> int:
    if UV is None:
        raise RuntimeError("uv is required to run the checker conformance suite")
    expected = {
        index
        for index, line in enumerate(FIXTURE.read_text().splitlines(), start=1)
        if line.endswith("# error")
    }
    failed = False
    for command in COMMANDS:
        result = subprocess.run(  # noqa: S603 - fixed repository commands, no shell
            [UV, "run", *command, str(FIXTURE)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        reported = {
            int(match) for match in re.findall(r"invalid\.py:(\d+):", result.stdout)
        }
        missing = expected - reported
        if result.returncode != 1 or missing:
            failed = True
            print(f"{command[0]}: FAILED; unreported invalid lines: {sorted(missing)}")
            print(result.stdout)
        else:
            print(f"{command[0]}: rejected all {len(expected)} invalid expressions")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
