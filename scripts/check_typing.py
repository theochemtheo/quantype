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
# A single invalid fixture needs no parallel workers (and no IPC status files).
COMMANDS = (
    (
        "mypy",
        "--strict",
        "--num-workers",
        "0",
        "--no-native-parser",
        "--cache-dir",
        ".mypy_cache/standard",
    ),
    (
        "mypy",
        "--strict",
        "--num-workers",
        "0",
        "--native-parser",
        "--cache-dir",
        ".mypy_cache/native",
    ),
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
        label = " ".join(command)
        result = subprocess.run(  # noqa: S603 - fixed repository commands, no shell
            [UV, "run", *command, str(FIXTURE)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        # Checkers colour their output when FORCE_COLOR is set, even into a pipe.
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        reported = {int(match) for match in re.findall(r"invalid\.py:(\d+):", output)}
        missing = expected - reported
        if result.returncode != 1 or missing:
            failed = True
            print(f"{label}: FAILED; unreported invalid lines: {sorted(missing)}")
            print(result.stdout)
        else:
            print(f"{label}: rejected all {len(expected)} invalid expressions")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
