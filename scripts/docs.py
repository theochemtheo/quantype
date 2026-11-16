"""Serve the production documentation artifact under its GitHub Pages path."""

import shutil
import tomllib
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


class ProjectPathHandler(SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/quantype":
            self.send_response(301)
            self.send_header("Location", "/quantype/")
            self.end_headers()
        elif path.startswith("/quantype/"):
            self.path = self.path.removeprefix("/quantype")
            super().do_GET()
        else:
            self.send_error(404, "Preview is mounted at /quantype/")


def preview() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    with TemporaryDirectory(prefix="quantype-docs-preview-") as temporary:
        artifact = Path(temporary) / "site"
        shutil.copytree(ROOT / config["site_dir"], artifact)
        handler = partial(ProjectPathHandler, directory=artifact)
        with ThreadingHTTPServer(("127.0.0.1", 8766), handler) as server:
            url = "http://127.0.0.1:8766/quantype/"
            print(f"Production preview: {url}", flush=True)
            webbrowser.open(url)
            server.serve_forever()


if __name__ == "__main__":
    preview()
