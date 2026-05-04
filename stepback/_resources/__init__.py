"""Bundled metadata resources shipped inside the wheel.

This subpackage exists *only* so the project ``LICENSE``, ``CITATION.cff``,
and ``README.md`` can be opened from an installed wheel via
``importlib.resources``. The canonical copies live at the repository root;
``scripts/sync_packaged_resources.py`` mirrors them in here, and
``tests/test_packaged_resources.py`` enforces that they stay byte-identical.

See :mod:`stepback.resources` for the public accessor API.
"""
