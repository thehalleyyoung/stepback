# Additional journal checkpoint, 2026-10-07

Prospective contract:plans/mutable-state-boundary.md. New local deterministic
experiment:study_audit/mutable_state_boundary.py,12 integer inputs and one changed
captured offset. Corrected result:results/audit/mutable-state-boundary-normalized/result.json.
Private same-instance and stale shared scopes yield 0/12 direct-equal values with
zero calls; semantic-version namespaces and the conventional versioned memoizer
yield 12/12 with 12 calls. Stable warm repetitions use zero calls. This boundary
confirms caller-state stability is required even in default private scopes.

First run:results/audit/mutable-state-boundary/result.json and matching log retain
an output-wrapper comparison failure. The pre-correction script is preserved in
versions/journal-round/. It is a harness failure, not evidence that versioned
Stepback execution produced wrong callback values. Normalization compares each
trace tool output's `result` with the uncached callback value. No observations
were dropped or relabeled as live tasks. No implementation change is justified:
mutable state without a declared version is outside the stable-semantics contract.

Title, abstract, central claim, comparison, conclusion and limitations now align:
matched prefix reuse; explicit content reconstruction and versioning obligations;
no incremental baseline benefit, no formal/Python refinement or task outcome claim.
Prior primary literature remains in audit/prior-art.md; this local comparison does
not establish a novel cache technique.97 focused cache tests passed in 0.37 s with
one stale-price warning; earlier complete-suite failures remain retained and no
final full-suite green claim is made. New spend$0; historical invoice reconciliation
remains absent. A scoped archival release is useful; journal readiness is NOT_YET.

All preexisting edited source bytes were preserved under versions/journal-round
(the earlier baseline and git versions also remain). PDF cleanup archives the
superseded PDF externally by SHA 256 before removal from this repository. The
GitHub Pages workflow stages a self-contained site and paper only when manually
invoked; no push, deploy or upload has occurred in this local work.

Final local verification:110 combined focused cache/resource tests passed in0.73s (one historical-price-age warning); compilation produced a 4-page PDF with no final undefined-reference or overfull warnings. Pages 1 and 4 were visually inspected before final whitespace correction and had no clipping. Both source-preview and staged HTML local links resolve. The temporary staging copy was removed, leaving only root tool_paper.pdf. Final build paths/failures remain logged; no full-suite rerun.
