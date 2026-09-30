"""Every Python example in the README and docs runs as written.

Blocks are read from the repository's own Markdown and executed in a fresh
namespace, with a temporary working directory so examples that save files
leave nothing behind. A fence written ```python notest marks an API pattern
that is not meant to run (placeholders such as ``data``); GitHub still
highlights it as Python.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
FENCE = re.compile(r"^```python([^\n]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
BACKENDS = ("jax", "torch")


def _examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for document in DOCUMENTS:
        text = document.read_text()
        for match in FENCE.finditer(text):
            if "notest" in match.group(1).split():
                continue
            line = text.count("\n", 0, match.start()) + 1
            examples.append((f"{document.relative_to(ROOT)}:{line}", match.group(2)))
    return examples


EXAMPLES = _examples()


@pytest.mark.parametrize(
    ("name", "source"), EXAMPLES, ids=[name for name, _ in EXAMPLES]
)
def test_example(
    name: str, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for backend in BACKENDS:
        if re.search(rf"^\s*(import|from) {backend}\b", source, re.MULTILINE):
            pytest.importorskip(backend)
    monkeypatch.chdir(tmp_path)
    code = compile(source, name, "exec")
    exec(code, {"__name__": "__docs__"})  # noqa: S102 -- the repository's own docs
