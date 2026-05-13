# stepback deprecation policy

This document is the canonical policy for changing or removing public
APIs in the `stepback` Python package. SB-Trace wire-format
compatibility is governed separately (see
[`spec/`](../spec) once it lands and `docs/canonicalization.md`).

The rules below apply to every name in
[`stepback.__all__`](../stepback/__init__.py) and to every name
re-exported from a `stepback.*` submodule that does not start with an
underscore.

## TL;DR

> **No public API may be removed or have its observable behaviour
> changed in a backwards-incompatible way without first shipping at
> least one minor release in which the old API still works but emits a
> `DeprecationWarning` that names the replacement and the planned
> removal release.**

## What counts as a public API

A symbol is **public** if any of the following is true:

* It appears in `stepback.__all__`.
* It is documented in `README.md`, in `docs/`, or in a published
  release announcement.
* It is named in `tests/test_public_api.py::EXPECTED_PUBLIC_API`.
* It is exported by a `stepback.*` submodule and does not begin with
  an underscore.

Everything else — including names beginning with an underscore,
modules in `stepback._*`, anything under `stepback.bench`,
`stepback.proxy`, or `stepback.testing.fixtures`, plus internal
attributes of public classes — is **private** and may change without
warning. If you want stable access to a private symbol, open an issue
and we will discuss promoting it.

## The cycle

A backwards-incompatible change goes through three distinct releases:

1. **Deprecate (release `N.M.0`).**
   * The old API keeps working unchanged.
   * The first call to it emits a single
     `DeprecationWarning` whose text is produced by
     `stepback.format_deprecation_message(...)`. The message must
     state both `since=N.M` and the `removal=` release.
   * The release notes for `N.M.0` add a `### Deprecated` heading to
     `CHANGELOG.md` listing the symbol, the replacement, and the
     planned removal.
   * The replacement API must already be present and tested in
     `N.M.0`; deprecating without a replacement is only allowed when
     the entire feature is being removed, in which case the warning
     must say so explicitly.

2. **Overlap (release `N.(M+1).0` or later).**
   * Both APIs continue to work.
   * Bug fixes apply to both.
   * Documentation is updated to use the replacement everywhere; the
     deprecated form survives only in the changelog and migration
     notes.

3. **Remove (release `removal`).**
   * The deprecated symbol is deleted.
   * `CHANGELOG.md` adds a `### Removed` heading mirroring the
     earlier `### Deprecated` entry, with a link to the migration
     advice.
   * `tests/test_public_api.py::EXPECTED_PUBLIC_API` is updated in
     the same commit.

The minimum overlap is one minor release. Long-lived deprecations
(more than one minor release of overlap) are encouraged for any API
that has shipped in a tagged release for more than three months.

## How to deprecate code

Use the helpers re-exported from the top-level package. Do not call
`warnings.warn(..., DeprecationWarning)` directly — the helpers
enforce the policy at import time and produce a uniform message that
release tooling can grep.

### Deprecate a function

```python
from stepback import deprecated

@deprecated(
    replacement="stepback.replay",
    since="0.2",
    removal="0.4",
)
def load_trace(path):
    ...
```

The decorator preserves `__wrapped__`, the original signature, and
the original docstring (with a `.. deprecated::` note appended). The
warning fires on every call but `warnings`'s default filter
de-duplicates per call site, so a typical agent run produces at most
one warning per deprecated symbol.

### Deprecate a class

```python
from stepback import deprecated

@deprecated(
    replacement="stepback.SubstitutionSet",
    since="0.2",
    removal="0.4",
)
class LegacySubstitutions:
    ...
```

The warning fires on construction.

### Rename a function or class

```python
from stepback import deprecated_alias

def new_compute_dirty_set(...): ...

# Keep the old name importable for one minor release.
compute_dirty_set = deprecated_alias(
    new_compute_dirty_set,
    name="compute_dirty_set",
    since="0.2",
    removal="0.4",
)
```

### Deprecate a parameter or behaviour change

When the symbol stays but its signature or semantics change, call
`warn_deprecated` from inside the function body so the warning fires
only on the affected code path:

```python
from stepback import warn_deprecated

def replay(path, *, strict=None):
    if strict is False:
        warn_deprecated(
            "stepback.replay(strict=False)",
            replacement="stepback.replay(strict=True) (the new default)",
            since="0.2",
            removal="0.4",
        )
    ...
```

## What downstream users must do

* **Run your test suite with deprecation warnings visible.** The
  pytest invocation `pytest -W error::DeprecationWarning` turns every
  stepback deprecation into a test failure, which is the easiest way
  to find affected call sites.
* **Pin a stepback minor version in production.** Each minor release
  may add deprecation warnings; pinning lets you update on your own
  schedule.
* **Read `CHANGELOG.md`** before bumping the minor version.

## Enforcement

`tests/test_deprecation.py` exercises the helpers, asserts that the
canonical message format does not regress, and confirms that calling
`format_deprecation_message` without a `since=` or `removal=` raises
`DeprecationPolicyError`. `tests/test_public_api.py` is the snapshot
that fails loudly when a public symbol disappears without going
through this cycle.

A new linter step in CI (planned alongside `### § API governance`
step 25 of the 100-steps roadmap) will diff the public surface
against the previous tag and refuse PRs that delete a public name
without a corresponding `### Deprecated` entry from at least one
prior release.
