"""Check built local targets and curated agent exports without network access."""

import json
import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from markdown import markdown as render_markdown

ROOT = Path(__file__).resolve().parents[1]
CODE_BLOCK = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
HISTORICAL = {"proposals", "design", "superseded", "_adr_template"}


class Targets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        for name in ("href", "src"):
            if value := attributes.get(name):
                self.links.append(value)


def markdown_links(text: str) -> list[str]:
    parsed = Targets()
    parsed.feed(render_markdown(text, extensions=["fenced_code", "tables"]))
    return parsed.links


def check_link(
    site: Path, base: str, pages: dict[Path, Targets], source: Path, link: str
) -> Path | None:
    origin = urlsplit(base)
    url = urlsplit(urljoin(urljoin(base, source.relative_to(site).as_posix()), link))
    if url.scheme not in {"http", "https"} or url.netloc != origin.netloc:
        return None
    if not url.path.startswith(origin.path):
        raise SystemExit(f"{source.relative_to(site)}: escaped project path: {link}")
    target = site / unquote(url.path.removeprefix(origin.path))
    if url.path.endswith("/"):
        target /= "index.html"
    if not target.is_file():
        raise SystemExit(f"{source.relative_to(site)}: missing target: {link}")
    html = target.with_suffix(".html") if target.suffix == ".md" else target
    if url.fragment and html in pages and unquote(url.fragment) not in pages[html].ids:
        raise SystemExit(f"{source.relative_to(site)}: missing fragment: {link}")
    return target


def check_markdown(
    site: Path, base: str, pages: dict[Path, Targets], source: Path
) -> None:
    markdown = source.read_text()
    relative = source.relative_to(site)
    if not markdown.startswith("# "):
        raise SystemExit(f"Missing exported title: {relative}")
    original = ROOT / "docs" / relative
    if relative.name == "index.md" and not original.is_file():
        original = ROOT / "docs" / relative.parent.with_suffix(".md")
    for block in CODE_BLOCK.findall(original.read_text()):
        if block.rstrip() not in markdown:
            raise SystemExit(f"Code lost from Markdown export: {relative}")
    for link in markdown_links(markdown):
        check_link(site, base, pages, source, link)


def check_exports(site: Path, base: str, pages: dict[Path, Targets]) -> int:
    index = site / "llms.txt"
    exports = set()
    for link in markdown_links(index.read_text()):
        target = check_link(site, base, pages, index, link)
        if target is None or target.suffix != ".md":
            raise SystemExit(f"Agent entry is not local Markdown: {link}")
        exports.add(target)
    if not exports:
        raise SystemExit("Agent index has no Markdown pages")
    for source in exports:
        check_markdown(site, base, pages, source)
    home = (site / "index.md").read_text()
    identity = r"Development documentation.*package \S+.*source [0-9a-f]{40}"
    if not re.search(identity, home):
        raise SystemExit("Exported homepage lacks build identity")
    return len(exports)


def check_search(site: Path) -> None:
    search = json.loads((site / "search.json").read_text())
    for item in search["items"]:
        path = Path(urlsplit(item["location"]).path)
        if HISTORICAL.intersection(path.parts) or path.stem == "_adr_template":
            raise SystemExit(f"Historical page in search: {path}")


def check() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    site = ROOT / config["site_dir"]
    base: str = config["site_url"]
    pages: dict[Path, Targets] = {}
    for path in site.rglob("*"):
        relative = path.relative_to(site)
        if HISTORICAL.intersection(relative.parts) or path.stem == "_adr_template":
            raise SystemExit(f"Historical output was published: {relative}")
        if path.suffix == ".html":
            parsed = Targets()
            parsed.feed(path.read_text())
            pages[path] = parsed
    for source, parsed in pages.items():
        for link in parsed.links:
            check_link(site, base, pages, source, link)
    count = check_exports(site, base, pages)
    check_search(site)
    print(f"Checked {len(pages)} HTML pages and {count} agent Markdown exports")


if __name__ == "__main__":
    check()
