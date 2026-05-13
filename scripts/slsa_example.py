#!/usr/bin/env python3
"""SLSA / in-toto provenance example for stepback traces.

This script demonstrates how to generate, sign, and verify SLSA Provenance
v1 / in-toto attestations for stepback ``.sb`` traces using the
:mod:`stepback.provenance` module.

It runs end-to-end without network access and without any extra dependencies
beyond ``stepback`` itself.

Usage
-----
.. code-block:: bash

    python scripts/slsa_example.py

    # Run with a custom trace file
    python scripts/slsa_example.py --trace /path/to/my_trace.sb

    # Emit the signed envelope to a file
    python scripts/slsa_example.py --out provenance.json

The script:

1. Records a small deterministic agent trace to a temporary ``.sb`` file.
2. Generates a SLSA Provenance v1 statement for the trace.
3. Signs the statement with a freshly-generated Ed25519 key.
4. Verifies the signature.
5. Prints the DSSE envelope as pretty-printed JSON.
6. Optionally writes the envelope to ``--out``.

SLSA / in-toto concepts
-----------------------
* **Statement** — a JSON document asserting facts about a *subject*
  (the ``.sb`` file) using a named *predicate* (here: SLSA Provenance v1).
* **DSSE envelope** — a simple signing wrapper: the statement bytes are
  base64-encoded and signed with Ed25519 via the DSSE PAE scheme.
* **SLSA Provenance v1** — a predicate that records the *builder*,
  *build type*, *invocation* parameters, and *materials* (input
  artefacts) that produced the subject.

For production use, replace the generated ``Ed25519PrivateKey`` with a
key from your CI secrets store, KMS, or YubiHSM.  See ``stepback hwkey``
and :mod:`stepback.hwkey` for hardware-backed signing.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Bootstrap path so this script works from the repo root without installation.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import stepback
from stepback.provenance import (
    trace_provenance,
    sign_provenance,
    verify_provenance_signature,
    sha256_of_file,
)
from stepback.testing import run_recorded_agent


def _record_example_trace(output_path: str) -> None:
    """Record a short deterministic agent run to *output_path*."""
    from stepback import record

    with record(output_path) as rec:
        run_recorded_agent(rec)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SLSA / in-toto provenance example for stepback traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trace",
        metavar="PATH",
        help="Path to an existing .sb trace file.  When omitted, a short "
             "example trace is recorded into a temporary file.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write the signed DSSE envelope to this path as JSON.",
    )
    parser.add_argument(
        "--builder-id",
        default="https://example.com/builder/dev",
        help="SLSA builder identifier URI (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Obtain a .sb trace
    # ------------------------------------------------------------------
    own_tmpfile: str | None = None
    trace_path = args.trace

    if trace_path is None:
        fd, own_tmpfile = tempfile.mkstemp(suffix=".sb", prefix="slsa_example_")
        os.close(fd)
        trace_path = own_tmpfile
        print(f"[1/5] Recording example trace → {trace_path}")
        _record_example_trace(trace_path)
    else:
        print(f"[1/5] Using existing trace  → {trace_path}")

    file_sha256 = sha256_of_file(trace_path)
    file_size = os.path.getsize(trace_path)
    print(f"      sha256={file_sha256}  size={file_size} bytes")

    # ------------------------------------------------------------------
    # 2. Generate an Ed25519 key pair (in production: use a real key)
    # ------------------------------------------------------------------
    print("[2/5] Generating ephemeral Ed25519 signing key …")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes_raw()
    key_id = f"ed25519:{public_key_bytes.hex()}"
    print(f"      key_id = {key_id[:48]}…")

    # ------------------------------------------------------------------
    # 3. Build the SLSA Provenance v1 statement
    # ------------------------------------------------------------------
    print("[3/5] Building SLSA Provenance v1 statement …")
    statement = trace_provenance(
        trace_path,
        sha256_digest=file_sha256,
        builder_id=args.builder_id,
    )
    print(f"      subject: {statement['subject'][0]['name']}")
    print(f"      builder: {statement['predicate']['runDetails']['builder']['id']}")
    print(f"      buildType: {statement['predicate']['buildDefinition']['buildType']}")

    # ------------------------------------------------------------------
    # 4. Sign the statement → DSSE envelope
    # ------------------------------------------------------------------
    print("[4/5] Signing with Ed25519 (DSSE/PAE) …")
    envelope = sign_provenance(statement, private_key)
    print(f"      payloadType: {envelope['payloadType']}")
    print(f"      sig (first 16 chars): {envelope['signatures'][0]['sig'][:16]}…")

    # ------------------------------------------------------------------
    # 5. Verify the signature
    # ------------------------------------------------------------------
    print("[5/5] Verifying signature …")
    verify_provenance_signature(envelope, public_key)
    print("      ✓ Signature verified successfully")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    envelope_json = json.dumps(envelope, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(envelope_json)
        print(f"\n✓ Envelope written to {args.out}")
    else:
        print("\n── DSSE Envelope ─────────────────────────────────────────────")
        # Decode the payload for human-readable display.
        payload_bytes = base64.b64decode(envelope["payload"])
        statement_pretty = json.dumps(json.loads(payload_bytes), indent=2)
        print("── Statement (decoded payload) ────────────────────────────────")
        print(statement_pretty)
        print("── Envelope JSON ──────────────────────────────────────────────")
        print(envelope_json)

    # Clean up temp file
    if own_tmpfile and os.path.exists(own_tmpfile):
        os.unlink(own_tmpfile)

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
