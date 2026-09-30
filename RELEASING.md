# Releasing

A release is cut by pushing a version tag. The `release.yml` workflow then
builds the wheel and sdist, publishes them to PyPI via trusted publishing, and
creates the GitHub Release — notes taken from this version's `CHANGELOG.md`
section, with the built artifacts attached. The tag is the only manual trigger.

## Steps

1. Open a release PR that:
   - bumps the version in `pyproject.toml` (`uv version X.Y.Z` also refreshes
     `uv.lock`);
   - adds a `CHANGELOG.md` section under a `## [X.Y.Z] - YYYY-MM-DD` heading,
     with a matching link reference at the foot of the file.
2. Preview the release notes: `scripts/changelog-section.sh X.Y.Z`.
3. Merge the PR.
4. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The workflow refuses a tag that does not match the version in `pyproject.toml`.

## Prerequisites

PyPI trusted publishing must be configured for the `quantype` project — owner
`theochemtheo`, repository `quantype`, workflow `release.yml`, environment
`pypi`.

## Versioning

quantype follows [Semantic Versioning](https://semver.org). Before 1.0.0, a
minor release may make breaking changes; each is recorded in `CHANGELOG.md`.
Changing the default CODATA edition is a minor release.
