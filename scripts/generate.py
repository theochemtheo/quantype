"""Regenerate/check the built-in API using the public catalogue generator."""

import argparse
from pathlib import Path

from quantype.catalogue import builtin_catalogue
from quantype.codegen import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "src/quantype"
    stale = generate(builtin_catalogue(), directory, check=args.check)
    if args.check and stale:
        parser.exit(1, "Stale generated files:\n" + "\n".join(stale) + "\n")


if __name__ == "__main__":
    main()
