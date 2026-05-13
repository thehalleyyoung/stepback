# Contributing to stepback

Thank you for your interest in contributing to **stepback**! This document
covers the mechanics of getting started, coding conventions, testing
requirements, and the pull-request process.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Ways to contribute](#ways-to-contribute)
3. [Development environment](#development-environment)
4. [Running the tests](#running-the-tests)
5. [Coding conventions](#coding-conventions)
6. [Commit messages and DCO](#commit-messages-and-dco)
7. [Opening an issue](#opening-an-issue)
8. [Submitting a pull request](#submitting-a-pull-request)
9. [Release process](#release-process)
10. [Reporting security issues](#reporting-security-issues)

---

## Code of conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

---

## Ways to contribute

* **Bug reports** — file a [bug report](https://github.com/stepback-dev/stepback/issues/new?template=bug_report.yml).
* **Feature requests** — open a [feature request](https://github.com/stepback-dev/stepback/issues/new?template=feature_request.yml).
* **Documentation improvements** — docs live in `docs/` and in docstrings.
* **Tests** — new replay scenarios, fixture traces, and edge-case coverage.
* **Bug fixes and features** — see the open issues for ideas.

---

## Development environment

Requires Python 3.10 or later.

```sh
# 1. Clone
git clone https://github.com/stepback-dev/stepback.git
cd stepback

# 2. Create a virtual environment (or use pipx / conda)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode with all dev dependencies
pip install -e ".[dev,shims,bench]"

# 4. Verify the suite passes
pytest -q --tb=short -m "not overhead_budget"
```

### Rust / multi-language bindings (optional)

See [`docs/INSTALL.md`](docs/INSTALL.md) for instructions on building the
Rust `stepback-core` crates, PyO3 bindings, TypeScript, Go, JVM, .NET, and
WASM targets.

---

## Running the tests

```sh
# Full suite (fast tests only — overhead-budget tests measure wall-clock time
# and are skipped in CI except on the nightly run)
pytest -q --tb=short -m "not overhead_budget"

# Include timing tests (may fail on a loaded machine)
pytest -q --tb=short

# Run a specific test file
pytest tests/test_replay.py -v

# Coverage
coverage run -m pytest -q --tb=short -m "not overhead_budget"
coverage report --show-missing
```

All pull requests must keep the test suite green (`pytest -q --tb=short -m "not overhead_budget"`).
New behaviour must be accompanied by new tests.  Where applicable, add a
conformance trace to `tests/fixtures/` so the behaviour is pin-tested
independently of the implementation.

---

## Coding conventions

* **Style** — PEP 8 with 88-character line length (Black-compatible).
* **Type annotations** — all public functions and methods must be annotated.
  Run `mypy stepback/` to check.
* **Docstrings** — Google-style; public API must have a one-line summary plus
  Args/Returns/Raises sections where non-trivial.
* **Imports** — `from __future__ import annotations` at the top of every
  module; stdlib before third-party before local.
* **Private helpers** — prefix with a single underscore.
* **No new runtime dependencies** without a discussion issue; the current
  runtime dependency surface is intentionally tiny (`cryptography`, `packaging`).

---

## Commit messages and DCO

This project uses the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).
Every commit must include a `Signed-off-by` trailer matching your real name
and email address:

```
git commit -s -m "feat(replay): add parallel-branch pruning"
```

The full DCO text is reproduced in [`meta/run_obligations.md`](meta/run_obligations.md).

Commit message format (loosely Conventional Commits):

```
<type>(<scope>): <short summary>

[optional body — wrap at 72 chars]

[optional footers]
Signed-off-by: Your Name <you@example.com>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`.

---

## Opening an issue

* Use one of the structured templates (bug report or feature request).
* **Security vulnerabilities** — do **not** open a public issue; see
  [SECURITY.md](SECURITY.md) for the coordinated-disclosure process.
* Search existing open issues before filing a duplicate.

---

## Submitting a pull request

1. Fork the repo and create a feature branch from `main`:
   ```sh
   git checkout -b feat/my-change
   ```
2. Make your changes and add or update tests.
3. Ensure `pytest -q --tb=short -m "not overhead_budget"` passes locally.
4. Ensure `mypy stepback/` has no new errors.
5. Update `CHANGELOG.md` under `[Unreleased]` with a concise entry.
6. Push your branch and open a PR against `main`.
7. Fill in the PR template checklist.
8. A maintainer will review within 7 business days; address any requested
   changes promptly.

PRs that break the public API (`stepback/__all__`) require a version bump and
a deprecation notice in the relevant module before removal (see
[`docs/deprecation.md`](docs/deprecation.md)).

---

## Release process

Releases are tag-driven. Maintainers only:

1. Bump `stepback/__version__.py` to the new SemVer.
2. Update `CHANGELOG.md` — move `[Unreleased]` entries under the new version
   heading with today's date.
3. Tag: `git tag -s v<VERSION> -m "Release v<VERSION>"` and push the tag.
4. The [`release.yml`](.github/workflows/release.yml) workflow builds the
   wheel and sdist, verifies them, publishes to PyPI via Trusted Publishing,
   and creates a GitHub release with the changelog excerpt as notes.

---

## Reporting security issues

See [SECURITY.md](SECURITY.md).  The short version: use GitHub Security
Advisories or the contact email — **never** a public issue.
