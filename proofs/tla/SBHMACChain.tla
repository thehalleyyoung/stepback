---------------------------- MODULE SBHMACChain ----------------------------
(*
 * TLA+ specification of the SB-Trace HMAC chain integrity protocol.
 *
 * The .sb format writes frames as an append-only log.  Each frame carries:
 *
 *   body      -- the frame content (canonical JSON bytes, modelled abstractly)
 *   prev_hmac -- HMAC of the preceding frame (ZERO_HMAC for frame 1)
 *   hmac      -- HMAC_SHA256(key, prev_hmac || canonical_json(body))
 *
 * In addition, each frame's `hmac` field is signed with Ed25519 (`sig`).
 * This spec focuses on the HMAC chain layer; Ed25519 is treated as a
 * secondary endorsement and is not modelled separately.
 *
 * === Threat model ===
 *
 * The adversary may modify the *body* of any frame on disk but CANNOT:
 *   - recompute a valid HMAC without the HMAC secret key  (A_mac)
 *   - find two (prev, body) pairs that hash to the same tag  (A_coll)
 *
 * This is the "ciphertext-only" or "body-substitution" threat model.
 * Key compromise or length-extension attacks are out of scope for this
 * spec.
 *
 * === Abstract HMAC model ===
 *
 * AbstractHMAC(prev, body) is modelled as the structured tuple
 *   <<HMACKey, prev, body>>
 * TLC's built-in value equality makes tuples injective over the finite
 * model domain, correctly instantiating A_coll.  The HMAC key is
 * embedded in HMACKey (a CONSTANT), instantiating A_mac: an adversary
 * who does not know HMACKey cannot produce a tuple that matches the
 * writer's output.
 *
 * === Note on canonicalization ===
 *
 * This spec assumes that the canonicalization function is deterministic
 * and injective for the modelled set of bodies.  Correctness of canonical
 * JSON itself is verified separately (see spec/rfcs/0002-canonicalization.md
 * and tests/test_canonical_*.py).
 *
 * === Note on truncation ===
 *
 * The invariants here cover body-substitution tamper evidence.  Suffix
 * truncation leaves the *prefix* chain valid; detection of truncation
 * requires an externally-anchored end commitment (the Merkle summary frame
 * and tail frame in .sb v1, specified in spec/sbtrace-v1.md §4.5).  That
 * anchor is outside the scope of this spec.
 *
 * === Properties proved ===
 *
 *   P1 (Integrity)       -- untampered on-disk log always validates
 *   P2 (TamperEvidence)  -- body-modified log never validates
 *   P3 (WriterCoherence) -- writer's own log is always self-consistent
 *
 * Run with TLC to model-check these properties on a finite state space:
 *   cd proofs/tla && tlc SBHMACChain -config SBHMACChain.cfg
 *)

EXTENDS Integers, Sequences, TLC

CONSTANTS
    MaxFrames,   \* upper bound on frames appended in this model run
    Bodies,      \* finite abstract set of possible frame bodies
    HMACKey      \* abstract HMAC secret key (single-key model)

ASSUME MaxFrames \in Nat /\ MaxFrames >= 1
ASSUME Bodies # {}

\* Initial chaining value (ZERO_HMAC = 32 zero bytes in the implementation).
ZeroHMAC == "zero_hmac"

\* Abstract injective HMAC over the finite model domain.
\* Injectivity follows from TLC tuple equality; the key is baked in so that
\* an adversary lacking HMACKey cannot forge a matching tag.
AbstractHMAC(prev, body) == <<HMACKey, prev, body>>

\* --------------------------------------------------------------------------
VARIABLES
    writer_log,  \* sequence of frames as produced by the honest writer
    disk_log,    \* on-disk state (may diverge from writer_log after tamper)
    tampered     \* TRUE iff at least one frame body has been adversarially modified

vars == <<writer_log, disk_log, tampered>>

\* --------------------------------------------------------------------------
\* Helper: expected prev_hmac for position i in a given log.
PrevHMAC(log, i) ==
    IF i = 1 THEN ZeroHMAC ELSE log[i-1].hmac

\* Helper: expected HMAC for position i.
ExpectedHMAC(log, i) ==
    AbstractHMAC(PrevHMAC(log, i), log[i].body)

\* A log validates iff every frame's stored prev_hmac and hmac are correct.
LogValid(log) ==
    Len(log) >= 1 /\
    \A i \in 1..Len(log) :
        /\ log[i].prev_hmac = PrevHMAC(log, i)
        /\ log[i].hmac      = ExpectedHMAC(log, i)

\* --------------------------------------------------------------------------
Init ==
    /\ writer_log = << >>
    /\ disk_log   = << >>
    /\ tampered   = FALSE

\* The honest writer appends one authentic frame.  The chain is maintained
\* by construction: prev = last hmac on disk (or ZeroHMAC for the first frame).
AppendFrame(body) ==
    LET prev  == IF Len(writer_log) = 0
                 THEN ZeroHMAC
                 ELSE writer_log[Len(writer_log)].hmac
        h     == AbstractHMAC(prev, body)
        frame == [body |-> body, prev_hmac |-> prev, hmac |-> h]
    IN
    /\ Len(writer_log) < MaxFrames
    /\ writer_log' = Append(writer_log, frame)
    /\ disk_log'   = Append(disk_log,   frame)
    /\ UNCHANGED tampered

\* The adversary replaces the body of frame i on disk.
\* The adversary CANNOT modify prev_hmac or hmac (no key ⇒ cannot forge a tag).
\* Precondition newBody # disk_log[i].body ensures a genuine change occurred.
TamperBodyOnly(i, newBody) ==
    /\ i \in 1..Len(disk_log)
    /\ newBody # disk_log[i].body
    /\ disk_log' = [disk_log EXCEPT ![i].body = newBody]
    /\ tampered' = TRUE
    /\ UNCHANGED writer_log

\* --------------------------------------------------------------------------
Next ==
    \/ \E body \in Bodies            : AppendFrame(body)
    \/ \E i \in 1..Len(disk_log),
          body \in Bodies            : TamperBodyOnly(i, body)

Spec == Init /\ [][Next]_vars

\* --------------------------------------------------------------------------
\* P1 (Integrity): when no tampering has occurred, the on-disk log validates.
\* Holds by construction: AppendFrame builds frames with correct HMACs.
IntegrityInvariant ==
    (~tampered /\ Len(disk_log) >= 1) => LogValid(disk_log)

\* P2 (TamperEvidence): once a body is modified by an adversary who cannot
\* recompute the HMAC, the disk log no longer validates.
\* Specifically: the tampered frame's stored hmac was produced from the
\* *original* body; after body substitution, ExpectedHMAC returns a different
\* tuple, so the stored hmac no longer matches.
TamperEvidenceInvariant ==
    (tampered /\ Len(disk_log) >= 1) => ~LogValid(disk_log)

\* P3 (WriterCoherence): the writer's own log is always self-consistent,
\* regardless of what the adversary does to disk_log.
WriterCoherenceInvariant ==
    Len(writer_log) = 0 \/
    \A i \in 1..Len(writer_log) :
        /\ writer_log[i].prev_hmac = PrevHMAC(writer_log, i)
        /\ writer_log[i].hmac      = ExpectedHMAC(writer_log, i)

=============================================================================
