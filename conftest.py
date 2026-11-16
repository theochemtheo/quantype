"""Collect README and published user-documentation examples with Sybil."""

import ast
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import pytest
from sybil import Document, Example, Region, Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser
from sybil.typing import Evaluator

pytest_plugins = ("pytester",)

ROOT = Path(__file__).resolve().parent
EXCLUDED_PARTS = frozenset(
    {
        "contributing",
        "contributor",
        "contributors",
        "decisions",
        "design",
        "history",
        "proposals",
        "templates",
        "skills",
        "src",
        "source",
        ".agents",
        ".claude",
    }
)
CONTRIBUTOR_PAGES = frozenset(
    {"architecture.md", "development.md", "design-principles.md"}
)
FENCE = re.compile(
    r"^(?P<prefix>[ \t]*(?:(?:>\s*)+|(?:[-*+]|\d+[.)])\s+)?)"
    r"(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$"
)
PYTHON_INFO = re.compile(r"\b(?:python|pycon|doctest)\b")
REQUIREMENT = re.compile(r"<!-- requires: (.*?) -->")
SKIP = re.compile(r"<!-- skip: (next|start|end)(.*?) -->")


def _nav_paths(nav: object) -> Iterator[str]:
    if isinstance(nav, str):
        yield nav
    elif isinstance(nav, list):
        for entry in cast("list[object]", nav):
            yield from _nav_paths(entry)
    elif isinstance(nav, dict):
        for entry in cast("dict[str, object]", nav).values():
            yield from _nav_paths(entry)
    else:
        raise TypeError("Navigation must contain strings, lists, or tables")


def published_documents(root: Path) -> tuple[Path, ...]:
    """Use explicit site navigation, never a recursive Markdown glob."""
    with (root / "zensical.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    docs_dir = (root / project.get("docs_dir", "docs")).resolve()
    if not docs_dir.is_relative_to(root):
        raise ValueError("Documentation directory must be inside the repository")
    pages = _nav_paths(project["nav"])
    documents = [root / "README.md"]
    for page in pages:
        if "://" in page or page.startswith("#"):
            continue
        relative = Path(page.split("#", 1)[0])
        if relative.suffix != ".md":
            continue
        path = (docs_dir / relative).resolve()
        if not path.is_relative_to(docs_dir):
            raise ValueError(
                f"Navigation page escapes the documentation directory: {page}"
            )
        if (
            EXCLUDED_PARTS.intersection(relative.parts)
            or relative.name in CONTRIBUTOR_PAGES
        ):
            continue
        documents.append(path)
    return tuple(dict.fromkeys(documents))


@dataclass(frozen=True)
class PythonFence:
    start: int
    end: int
    requirements: tuple[str, ...]


def _syntax_error(document: Document, position: int, message: str) -> ValueError:
    return ValueError(f"{document.path}:{document.line_column(position)}: {message}")


def _check_skip(document: Document, position: int, text: str) -> None:
    skip = SKIP.fullmatch(text)
    if skip is None:
        raise _syntax_error(document, position, "Skip directives must be top-level")
    action, reason = skip.groups()
    if action == "end":
        if reason.strip():
            raise _syntax_error(document, position, "skip: end does not take a reason")
        return
    try:
        explanation = ast.literal_eval(reason.strip())
    except (SyntaxError, ValueError) as error:
        raise _syntax_error(
            document, position, "Skip directives need a quoted reason"
        ) from error
    if not isinstance(explanation, str) or not explanation.strip():
        raise _syntax_error(
            document, position, "Skip directives need a nonempty reason"
        )


def _check_directives(
    document: Document, position: int, text: str, following: str
) -> None:
    if "<!-- requires:" in text:
        if REQUIREMENT.fullmatch(text) is None:
            raise _syntax_error(document, position, "Malformed backend requirement")
        match = FENCE.fullmatch(following.rstrip("\r\n"))
        if match is None or match["info"].strip() != "python":
            raise _syntax_error(
                document, position, "requires must immediately precede a Python fence"
            )
    if "<!-- skip:" in text:
        _check_skip(document, position, text)
    if re.search(r"<!--\s*(?:invisible-)?code(?:-block)?:?\s+python\b", text):
        raise _syntax_error(
            document, position, "Python examples must use visible top-level fences"
        )


type _Opening = tuple[int, str, bool, tuple[str, ...]]


def _open_fence(
    document: Document, position: int, match: re.Match[str], previous: str
) -> _Opening:
    info = match["info"].strip()
    is_python = PYTHON_INFO.search(info) is not None
    if is_python and (info != "python" or match["prefix"]):
        raise _syntax_error(
            document, position, "Use an unannotated top-level python fence"
        )
    requirements: tuple[str, ...] = ()
    preceding = REQUIREMENT.fullmatch(previous)
    if is_python and preceding:
        requirements = tuple(part.strip() for part in preceding[1].split(","))
        if any(name not in {"jax", "torch"} for name in requirements):
            raise _syntax_error(
                document, position, "requires supports only jax and torch"
            )
    return position, match["fence"], is_python, requirements


def _continue_fence(
    document: Document, position: int, opening: _Opening, match: re.Match[str]
) -> tuple[_Opening | None, PythonFence | None]:
    start, original, is_python, requirements = opening
    marker, info = match["fence"], match["info"].strip()
    if marker[0] == original[0] and len(marker) >= len(original) and not info:
        if is_python:
            if match["prefix"]:
                raise _syntax_error(
                    document, position, "Python fences must close at top level"
                )
            return None, PythonFence(start, position + len(match.string), requirements)
        return None, None
    if PYTHON_INFO.search(info):
        raise _syntax_error(document, position, "Nested Python fences are unsupported")
    return opening, None


def python_fences(document: Document) -> list[PythonFence]:
    """Validate the supported authoring subset before Sybil can ignore anything."""
    fences: list[PythonFence] = []
    opening: _Opening | None = None
    previous = ""
    offset = 0
    lines = document.text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        text = line.rstrip("\r\n")
        match = FENCE.fullmatch(text)
        if match:
            if opening is None:
                opening = _open_fence(document, offset, match, previous)
            else:
                opening, fence = _continue_fence(document, offset, opening, match)
                if fence is not None:
                    fences.append(fence)
        elif opening is None:
            following = lines[index + 1] if index + 1 < len(lines) else ""
            _check_directives(document, offset, text, following)
        previous = text
        offset += len(line)
    if opening is not None and opening[2]:
        raise _syntax_error(document, opening[0], "Unclosed Python fence")
    return fences


def _fence_evaluator(
    evaluator: Evaluator,
    fence: PythonFence,
    started: set[int],
    *,
    shared: bool,
) -> Evaluator:

    def evaluate(example: Example) -> str | None:
        example.namespace.pop("_documentation_cwd", None)
        if fence.start not in started:
            if not shared:
                example.namespace.clear()
            started.add(fence.start)
        for backend in fence.requirements:
            if find_spec(backend) is None:
                pytest.importorskip(backend, exc_type=ModuleNotFoundError)
        return evaluator(example)

    return evaluate


def documentation_examples(document: Document) -> Iterator[Region]:
    """Parse Python once; preserve state between prompts, reset between fences."""
    fences = python_fences(document)
    shared = "tutorials" in Path(document.path).parent.parts
    regions = list(PythonCodeBlockParser()(document))
    started: set[int] = set()
    for fence in fences:
        enclosed = [
            region for region in regions if fence.start <= region.start < fence.end
        ]
        if not enclosed:
            raise _syntax_error(
                document, fence.start, "Python fence was not collected by Sybil"
            )
        for region in enclosed:
            evaluator = region.evaluator
            if evaluator is None:
                raise _syntax_error(
                    document, region.start, "Python example has no evaluator"
                )
            region.evaluator = _fence_evaluator(
                evaluator,
                fence,
                started,
                shared=shared,
            )
            yield region
    if any(
        not any(fence.start <= region.start < fence.end for fence in fences)
        for region in regions
    ):
        raise _syntax_error(
            document, 0, "Sybil collected Python outside a supported fence"
        )
    yield from SkipParser()(document)


DOCUMENTS = published_documents(ROOT)
SYBIL = Sybil(
    parsers=[documentation_examples],
    patterns=[str(path.relative_to(ROOT)) for path in DOCUMENTS],
    path=str(ROOT),
    fixtures=["_documentation_cwd"],
)
pytest_collect_file = SYBIL.pytest()


@pytest.fixture(scope="module")
def _documentation_cwd(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """One pytest-managed directory per document, including sequential file examples."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path_factory.mktemp("documentation"))
        yield
