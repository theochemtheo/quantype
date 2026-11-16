"""Expose the resolved build label in rendered and exported Markdown."""

import os
import subprocess
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any


@cache
def build_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        text=True,
    ).strip()
    kind = (
        "CI revision"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else "local working tree"
    )
    return (
        f"Development documentation · package {version('quantype')} · "
        f"source {revision} · {kind} (not a PyPI release)"
    )


def define_env(env: Any) -> None:  # noqa: ANN401
    env.variables["build_identity"] = build_identity()
