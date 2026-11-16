"""The README's links remain usable when rendered outside GitHub."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


REPOSITORY = "https://github.com/theochemtheo/quantype/blob/main/"
LINK = re.compile(r"\]\(([^)\s]+)\)")


def test_readme_links_work_off_github() -> None:
    """PyPI shows the README without the repository, so links are absolute."""
    for target in LINK.findall((ROOT / "README.md").read_text()):
        assert target.startswith(("https://", "#")), target
        if target.startswith(REPOSITORY):
            path = target.removeprefix(REPOSITORY).split("#", 1)[0]
            assert (ROOT / path).exists(), target
