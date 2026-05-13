"""Thin wrapper around :mod:`stepback.bench.soak`.

The canonical entry point is now ``stepback bench soak``; this
script is provided for shell pipelines and the scheduled
``.github/workflows/soak.yml`` workflow that prefers a stable file
path.
"""
from __future__ import annotations

import argparse
import json

from stepback.bench.soak import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-traces", type=int, default=10_000)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-substitution", action="store_true")
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--track-memory", action="store_true")
    p.add_argument("--out", help="write aggregate JSON to this path")
    args = p.parse_args()
    res = run(
        n_traces=args.n_traces,
        n_steps=args.n_steps,
        seed=args.seed,
        do_substitution=not args.no_substitution,
        progress_every=args.progress_every,
        track_memory=args.track_memory,
    )
    print(res.summary_line())
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res.to_json(), f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
