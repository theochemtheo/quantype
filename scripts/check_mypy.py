"""Run both mypy parsers with parallel workers and isolated caches."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".mypy_cache"
CONFIG = ROOT / "pyproject.toml"


def main() -> int:
    files = tomllib.loads(CONFIG.read_text())["tool"]["mypy"]["files"]
    CACHE.mkdir(exist_ok=True)
    parsers = (("standard", "--no-native-parser"), ("native", "--native-parser"))
    for parser, flag in parsers:
        command = [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(CONFIG),
            flag,
            "--cache-dir",
            str(CACHE / parser),
            *(str(ROOT / path) for path in files),
        ]
        # Parallel mypy writes .mypy_worker.*.json IPC status files in its cwd,
        # independently of --cache-dir. Keep those transient files out of ROOT.
        result = subprocess.run(command, cwd=CACHE, check=False)  # noqa: S603
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
