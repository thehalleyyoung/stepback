"""Thin wrapper around :mod:`stepback.bench.replay_caching`.

The canonical entry point is now ``stepback bench replay-caching``;
this script is kept for backwards compatibility with shell pipelines
that referenced it before the bench package landed.
"""
from __future__ import annotations

import argparse

from stepback.bench.replay_caching import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument(
        "--strategy",
        choices=["random_step", "first_quarter", "last_quarter",
                 "prompt_only", "tool_only"],
        default="random_step",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = run(
        n_steps=args.n_steps,
        n_trials=args.n_trials,
        strategy=args.strategy,
        seed=args.seed,
    )
    print(res.summary_line())


if __name__ == "__main__":
    main()
