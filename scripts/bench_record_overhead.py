"""Thin wrapper around :mod:`stepback.bench.record_overhead`.

The canonical entry point is now ``stepback bench record-overhead``;
this script is kept for backwards compatibility.
"""
from __future__ import annotations

import argparse

from stepback.bench.record_overhead import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-steps", type=int, default=1000)
    args = p.parse_args()
    res = run(n_steps=args.n_steps)
    print(res.summary_line())


if __name__ == "__main__":
    main()
