"""Regression contracts for Markdown collection and example state."""

import sys
from pathlib import Path
from unittest import SkipTest
from unittest.mock import Mock

import pytest
from sybil import Document
from sybil.example import SybilFailure

from conftest import ROOT, documentation_examples, published_documents


def _document(tmp_path: Path, source: str, page: str = "README.md") -> Document:
    path = tmp_path / page
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return Document.parse(str(path), documentation_examples)


def _evaluate(document: Document) -> None:
    for example in document.examples():
        example.evaluate()


@pytest.mark.parametrize(
    "page",
    [
        "README.md",
        "docs/how-to/task.md",
        "docs/reference/api.md",
        "docs/explanation/model.md",
        "docs/troubleshooting.md",
    ],
)
def test_independent_fences_do_not_leak_names(tmp_path: Path, page: str) -> None:
    document = _document(
        tmp_path,
        "```python\nsecret = 42\n```\n\n"
        "```python\nassert 'secret' not in globals()\n```\n",
        page,
    )
    _evaluate(document)


def test_doctest_prompts_share_state_only_within_the_fence(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        "```python\n>>> value = 40\n>>> value += 2\n>>> value\n42\n```\n\n"
        "```python\n>>> 'value' in globals()\nFalse\n```\n",
    )
    assert len(list(document.examples())) == 4
    _evaluate(document)


def test_tutorial_fences_share_sequential_state(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        "```python\nvalue = 40\n```\n\n"
        "```python\n>>> value += 2\n>>> value\n42\n```\n\n"
        "```python\nassert value == 42\n```\n",
        "docs/tutorials/nested/start.md",
    )
    assert len(list(document.examples())) == 4
    _evaluate(document)
    other = _document(
        tmp_path,
        "```python\nassert 'value' not in globals()\n```\n",
        "docs/tutorials/other.md",
    )
    _evaluate(other)


def test_doctest_output_mismatch_fails(tmp_path: Path) -> None:
    document = _document(tmp_path, "```python\n>>> 6 * 7\n41\n```\n")
    with pytest.raises(SybilFailure, match="Expected:"):
        _evaluate(document)


def test_no_user_imports_are_injected(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        "```python\n"
        "assert not {'np', 'pytest', 'quantype', 'u'} & globals().keys()\n```\n",
        "docs/tutorials/start.md",
    )
    _evaluate(document)


def test_requirements_skip_only_the_annotated_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importorskip = Mock(side_effect=pytest.skip.Exception("jax is missing"))
    monkeypatch.setattr(pytest, "importorskip", importorskip)
    monkeypatch.setattr("conftest.find_spec", Mock(return_value=None))
    document = _document(
        tmp_path,
        "<!-- requires: jax -->\n```python\n>>> 1\n1\n>>> 2\n2\n```\n\n"
        "```python\nassert True\n```\n",
    )
    examples = list(document.examples())
    for example in examples[:2]:
        with pytest.raises(pytest.skip.Exception, match="jax is missing"):
            example.evaluate()
    examples[2].evaluate()
    assert importorskip.call_count == 2
    importorskip.assert_called_with("jax", exc_type=ModuleNotFoundError)


def test_available_comma_separated_requirements_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importorskip = Mock()
    find_spec = Mock(return_value=object())
    monkeypatch.setattr(pytest, "importorskip", importorskip)
    monkeypatch.setattr("conftest.find_spec", find_spec)
    document = _document(
        tmp_path,
        "<!-- requires: jax, torch -->\n"
        "```python\nanswer = 42\nassert answer == 42\n```\n",
    )
    _evaluate(document)
    assert [call.args[0] for call in find_spec.call_args_list] == ["jax", "torch"]
    importorskip.assert_not_called()
    assert document.namespace["answer"] == 42


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_missing_requirement_uses_pytest_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    monkeypatch.setitem(sys.modules, backend, None)
    document = _document(
        tmp_path,
        f"<!-- requires: {backend} -->\n"
        "```python\nraise AssertionError('not executed')\n```\n",
    )
    with pytest.raises(pytest.skip.Exception):
        _evaluate(document)


def test_backend_imports_without_requirement_are_not_inferred(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        "```python\nimport quantype_documentation_missing_backend\n```\n",
    )
    with pytest.raises(
        ModuleNotFoundError, match="quantype_documentation_missing_backend"
    ):
        _evaluate(document)


def test_reasoned_sybil_skip_skips_illustrative_pattern(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        '<!-- skip: next "Illustrative application-provided data" -->\n'
        "```python\napplication_data.save()\n```\n\n"
        "```python\nassert True\n```\n",
    )
    examples = list(document.examples())
    examples[0].evaluate()
    with pytest.raises(SkipTest, match="Illustrative application-provided data"):
        examples[1].evaluate()
    examples[2].evaluate()


def test_skipped_first_prompt_still_resets_the_fence_namespace(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        "```python\nsecret = 42\n```\n"
        '<!-- skip: next "First prompt is illustrative" -->\n'
        "```python\n>>> application_data\n>>> 'secret' in globals()\nFalse\n```\n",
    )
    examples = list(document.examples())
    examples[0].evaluate()
    examples[1].evaluate()
    with pytest.raises(SkipTest, match="First prompt is illustrative"):
        examples[2].evaluate()
    examples[3].evaluate()


def test_reasoned_skip_range_handles_a_whole_transcript(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        '<!-- skip: start "Application-specific objects" -->\n'
        "```python\n>>> application_data\n...\n>>> application_result\n...\n```\n"
        "<!-- skip: end -->\n"
        "```python\nassert True\n```\n",
    )
    examples = list(document.examples())
    examples[0].evaluate()
    for example in examples[1:3]:
        with pytest.raises(SkipTest, match="Application-specific objects"):
            example.evaluate()
    for example in examples[3:]:
        example.evaluate()


@pytest.mark.parametrize(
    "source",
    [
        "```python notest\npass\n```\n",
        "```python title=example\npass\n```\n",
        "```{.python}\npass\n```\n",
        "```pycon\n>>> 1\n1\n```\n",
        "```doctest\n>>> 1\n1\n```\n",
        "    ```python\n    pass\n    ```\n",
        "> ```python\n> pass\n> ```\n",
        "- ```python\n  pass\n  ```\n",
        "````markdown\n```python\npass\n```\n````\n",
        "```python\npass\n",
        "<!-- requires: numpy -->\n```python\npass\n```\n",
        "<!-- requires: jax -->\n\n```python\npass\n```\n",
        "<!-- requires: jax -->\n",
        "<!-- skip: next -->\n```python\npass\n```\n",
        '<!-- skip: next "" -->\n```python\npass\n```\n',
        "<!-- invisible-code-block: python\nsecret = 1\n-->\n",
    ],
)
def test_unsupported_or_silently_missed_syntax_is_rejected(
    tmp_path: Path, source: str
) -> None:
    with pytest.raises(ValueError, match=r"fence|directives|requires|Python"):
        _document(tmp_path, source)


@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_supported_plain_and_repl_fences_are_collected_once(
    tmp_path: Path, marker: str
) -> None:
    document = _document(
        tmp_path,
        f"{marker}python\nassert 1 + 1 == 2\n{marker}\n\n"
        f"{marker}python\n>>> value = 2\n>>> value\n2\n{marker}\n",
    )
    assert len(list(document.examples())) == 3
    _evaluate(document)


def test_navigation_selects_nested_published_pages_only(tmp_path: Path) -> None:
    (tmp_path / "zensical.toml").write_text(
        "[project]\nnav = [\n"
        '  { Home = "index.md" },\n'
        '  { Tutorials = [{ First = "tutorials/first.md" }] },\n'
        '  { Reference = [{ API = "reference/nested/api.md" }] },\n'
        '  { Contributors = [{ Setup = "contributing/development.md" }] },\n'
        '  { Decisions = "decisions/ADR-001.md" },\n'
        '  { History = "history/old.md" },\n'
        '  { Source = "source/example.md" },\n'
        '  { Skills = "skills/authoring.md" },\n'
        '  { External = "https://example.com/guide.md" },\n'
        '  { Duplicate = "index.md" },\n'
        "]\n",
        encoding="utf-8",
    )
    assert published_documents(tmp_path) == tuple(
        tmp_path / page
        for page in (
            "README.md",
            "docs/index.md",
            "docs/tutorials/first.md",
            "docs/reference/nested/api.md",
        )
    )


def test_navigation_cannot_escape_docs(tmp_path: Path) -> None:
    (tmp_path / "zensical.toml").write_text(
        '[project]\nnav = [{ Source = "../src/example.md" }]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        published_documents(tmp_path)


def test_pytest_collection_uses_temporary_cwd_and_restores_it(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest((ROOT / "conftest.py").read_text(encoding="utf-8"))
    pytester.makefile(".md", README="```python\nassert True\n```\n")
    page = pytester.path / "docs/tutorials/start.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "```python\nfrom pathlib import Path\n"
        f"assert Path.cwd() != Path({str(pytester.path)!r})\n"
        "Path('example.txt').write_text('temporary')\n```\n\n"
        "```python\nassert Path('example.txt').read_text() == 'temporary'\n"
        "assert '_documentation_cwd' not in globals()\n```\n",
        encoding="utf-8",
    )
    pytester.makefile(
        ".toml",
        zensical='[project]\nnav = [{ Tutorial = "tutorials/start.md" }]\n',
    )
    pytester.makepyfile(
        test_after="from pathlib import Path\n"
        "def test_restored():\n"
        f"    assert Path.cwd() == Path({str(pytester.path)!r})\n",
    )
    result = pytester.runpytest("README.md", "docs", "test_after.py", "-q")
    result.assert_outcomes(passed=4)
    assert not (pytester.path / "example.txt").exists()
    assert not (page.parent / "example.txt").exists()
