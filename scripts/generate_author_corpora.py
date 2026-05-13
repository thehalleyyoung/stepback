#!/usr/bin/env python3
"""Generate (or regenerate) the author-original corpus ``.sb`` files.

Usage::

    python scripts/generate_author_corpora.py

Writes files to ``stepback/bench/corpora/`` (the package-bundled location).
Pass ``--force`` to overwrite existing ``.sb`` files; omit it to skip tasks
whose ``.sb`` file already exists (idempotent by default).

The generated files are deterministic in their HMAC keys (derived from a
fixed seed) but NOT byte-identical across runs because ``.sb`` frames embed
real wallclock timestamps.  Commit the generated files once; consumers should
rely on verify_trace(path, hmac_key) from manifest.json rather than byte
equality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from stepback.bench.author_corpora import generate_author_corpora  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "stepback" / "bench" / "corpora"),
        help="Directory to write corpus files (default: stepback/bench/corpora/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .sb files (default: skip if already present)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Generating author corpora → {output_dir}")
    generate_author_corpora(output_dir, force=args.force)

    # Print a summary
    for corpus_id in ["support-agent", "code-review", "payments-policy"]:
        corpus_dir = output_dir / corpus_id
        sb_files = list(corpus_dir.glob("*.sb"))
        print(f"  {corpus_id}: {len(sb_files)} .sb files")

    print("Done.")


if __name__ == "__main__":
    main()
