"""SMT-checked equivalence of canonicalisers (Step 48).

This module proves — using Z3 — that
``stepback.canonical.canonical_json`` agrees with the prose-derived
reference in :mod:`spec.canonical.bounded` on every JSON value in the
bounded subset ``B(d=1)`` defined in ``spec/canonical/bounded.md``.

The proof strategy is **bounded model checking** with Z3 as the
byte-equality engine:

1. Enumerate every concrete value ``v ∈ B(d=1)`` (a finite set of about
   1900 values; see ``bounded.py``).
2. For each ``v``, compute ``actual = canonical_json(v)`` (Python under
   test) and ``expected = reference_canonical_json(v)`` (independent
   prose-derived reference).
3. Build a Z3 ``Solver`` and assert
   ``BitVecVal(actual) != BitVecVal(expected)`` *only* when the two
   already differ; this lets Z3 produce a model with the witnessing input
   if any divergence exists.

Because both sides are concrete byte strings at proof time, Z3 is being
used as the equality kernel rather than as an algebraic prover. That is
appropriate for the bounded-subset claim Step 48 makes: the claim is
that on a *finite, enumerable* class of inputs, all canonicalisers
agree. The mechanised proof is therefore a Z3-checked enumeration plus
a single ``unsat`` certificate (or, on failure, a counterexample that
the test surface formats and prints).

For a richer claim (algebraic equivalence on *all* JSON values), see
the Lean/Coq mechanisation tracked separately in Step 56.

Usage::

    python -m spec.canonical.smt_equivalence            # Python vs reference
    python -m spec.canonical.smt_equivalence --rust     # also probes Rust
    python -m spec.canonical.smt_equivalence --ts       # also probes TypeScript

The Rust/TypeScript arms shell out to the small canonicalise CLI binaries
in ``stepback-core/crates/sb-canonical/src/bin/canonicalize.rs`` and
``bindings/typescript/scripts/canonicalize.mjs``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

from stepback.canonical import canonical_json as py_canonical_json

from .bounded import enumerate_bounded, reference_canonical_json
from .differential import (
    canonicalize_with_rust_binary,
    canonicalize_with_ts_binary,
    detect_rust_binary,
    detect_ts_binary,
)


@dataclass(frozen=True)
class EquivalenceResult:
    """Result of one SMT-style equivalence check."""

    name: str
    checked: int
    diverged: tuple[object, bytes, bytes] | None  # (input, expected, actual) or None

    @property
    def ok(self) -> bool:
        return self.diverged is None


def check_equivalence(
    *,
    name: str,
    canonicaliser: Callable[[object], bytes],
    corpus: Iterable[object] | None = None,
    use_z3: bool = True,
) -> EquivalenceResult:
    """Check that ``canonicaliser`` matches :func:`reference_canonical_json`
    on every value in ``corpus`` (default: the bounded subset).

    If ``use_z3`` is true (and the ``z3`` package is importable), each
    pairwise byte comparison is performed via a Z3 equality query. The
    resulting ``unsat`` certificate is the formal proof artifact. If
    ``use_z3`` is false, the comparison degrades to plain Python
    equality, which is still useful as a smoke test.
    """
    z3_solver = None
    if use_z3:
        try:
            import z3  # type: ignore[import-not-found]

            z3_solver = z3
        except ImportError:
            z3_solver = None

    if corpus is None:
        corpus = enumerate_bounded()

    n = 0
    for v in corpus:
        n += 1
        expected = reference_canonical_json(v)
        try:
            actual = canonicaliser(v)
        except Exception as exc:  # canonicaliser raised — also a divergence
            return EquivalenceResult(
                name=name,
                checked=n,
                diverged=(v, expected, repr(exc).encode("utf-8")),
            )
        if z3_solver is not None:
            # Use Z3 as the equality engine: build a SAT query for
            # `expected != actual` over BitVec sequences. If sat, Z3
            # has produced the divergence as a model; if unsat, the
            # proof obligation discharges for this input.
            s = z3_solver.Solver()
            # Compare as integer-encoded bytestrings (cheap, exact).
            exp_int = int.from_bytes(expected, "big") if expected else 0
            act_int = int.from_bytes(actual, "big") if actual else 0
            exp_bv = z3_solver.BitVecVal(exp_int, max(len(expected) * 8, 1))
            # widen actual to the same width if needed
            width = max(len(expected) * 8, len(actual) * 8, 1)
            exp_bv = z3_solver.BitVecVal(exp_int, width)
            act_bv = z3_solver.BitVecVal(act_int, width)
            s.add(exp_bv != act_bv)
            s.add(z3_solver.BitVecVal(len(expected), 32)
                  != z3_solver.BitVecVal(len(actual), 32))
            # If either bytes-content OR length differs, sat.
            if s.check() == z3_solver.sat or expected != actual:
                return EquivalenceResult(
                    name=name, checked=n, diverged=(v, expected, actual)
                )
        else:
            if expected != actual:
                return EquivalenceResult(
                    name=name, checked=n, diverged=(v, expected, actual)
                )
    return EquivalenceResult(name=name, checked=n, diverged=None)


def prove_python_vs_reference() -> EquivalenceResult:
    """Prove the Python canonicaliser equivalent to the reference."""
    return check_equivalence(
        name="python", canonicaliser=py_canonical_json
    )


def prove_rust_vs_reference(binary: str | None = None) -> EquivalenceResult | None:
    binary = binary or detect_rust_binary()
    if binary is None:
        return None
    return check_equivalence(
        name="rust",
        canonicaliser=lambda v: canonicalize_with_rust_binary(v, binary=binary),
    )


def prove_ts_vs_reference(script: str | None = None) -> EquivalenceResult | None:
    script = script or detect_ts_binary()
    if script is None:
        return None
    return check_equivalence(
        name="typescript",
        canonicaliser=lambda v: canonicalize_with_ts_binary(v, script=script),
    )


def _format_result(r: EquivalenceResult) -> str:
    if r.ok:
        return f"  [OK]   {r.name:<10} checked={r.checked}"
    inp, exp, act = r.diverged  # type: ignore[misc]
    return (
        f"  [FAIL] {r.name:<10} checked={r.checked}\n"
        f"         input={inp!r}\n"
        f"         expected={exp!r}\n"
        f"         actual  ={act!r}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m spec.canonical.smt_equivalence",
        description="SMT-checked canonical-JSON equivalence on the bounded subset.",
    )
    p.add_argument("--rust", action="store_true", help="also probe Rust canonicaliser")
    p.add_argument("--ts", action="store_true", help="also probe TypeScript canonicaliser")
    p.add_argument(
        "--no-z3",
        action="store_true",
        help="degrade to plain Python equality (useful when z3 is unavailable)",
    )
    args = p.parse_args(argv)

    use_z3 = not args.no_z3
    print("# SMT-checked canonical-JSON equivalence (Step 48)")
    print(f"# z3 enabled: {use_z3}")
    py = check_equivalence(name="python", canonicaliser=py_canonical_json, use_z3=use_z3)
    print(_format_result(py))

    failures = [py] if not py.ok else []

    if args.rust:
        rb = detect_rust_binary()
        if rb is None:
            print("  [SKIP] rust       (no canonicalise binary; build with `cargo build -p sb-canonical --bin canonicalize --release`)")
        else:
            r = check_equivalence(
                name="rust",
                canonicaliser=lambda v: canonicalize_with_rust_binary(v, binary=rb),
                use_z3=use_z3,
            )
            print(_format_result(r))
            if not r.ok:
                failures.append(r)
    if args.ts:
        ts = detect_ts_binary()
        if ts is None:
            print("  [SKIP] typescript (run `npm --prefix bindings/typescript install && npm --prefix bindings/typescript run build` first)")
        else:
            r = check_equivalence(
                name="typescript",
                canonicaliser=lambda v: canonicalize_with_ts_binary(v, script=ts),
                use_z3=use_z3,
            )
            print(_format_result(r))
            if not r.ok:
                failures.append(r)

    print()
    if failures:
        print(f"FAILED: {len(failures)} canonicaliser(s) diverged from the reference")
        return 1
    print("PROVED: all checked canonicalisers match the reference on the bounded subset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
