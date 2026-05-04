# SB-Trace conformance fixtures

This directory ships with the `stepback` wheel and sdist as
`stepback.conformance.fixtures`. It holds the frozen `.sb` traces that
every independent implementation of the SB-Trace format must read,
verify, canonicalize, and (for corrupt variants) reject.

The fixture corpus is populated by Step 41 of `100_STEPS.md`. Until then
this directory intentionally contains only this README so that the
package-data rule has something to ship and the
`stepback.conformance.iter_fixtures()` accessor remains usable.

To enumerate fixtures from Python:

```python
from stepback.conformance import iter_fixtures
for path in iter_fixtures():
    ...
```
