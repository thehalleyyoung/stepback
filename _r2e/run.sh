#!/usr/bin/env bash
# r2e_gym harness for slot=infer.
#
# Exercises stepback's primary "infer" path: record a deterministic
# fixture agent run, then REPLAY the trace (zero-LLM inference)
# under three conditions and assert measurable behaviour.
#
# Prints exactly one JSON line on stdout (last non-empty line).
set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 2
ROOT="$(pwd)"
LOG="$ROOT/_r2e/_run.log"
HIST="$ROOT/_r2e/_history.jsonl"
RESULT="$ROOT/_r2e/result.json"
: > "$LOG"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

# --- 1. Run the existing e2e replay test (the canonical infer flow).
"$PY" -m pytest tests/test_e2e_replay.py -q --no-header \
    >>"$LOG" 2>&1
PYTEST_RC=$?

E2E_PASSED=$(grep -Eo '[0-9]+ passed' "$LOG" | tail -1 | awk '{print $1}')
E2E_FAILED=$(grep -Eo '[0-9]+ failed' "$LOG" | tail -1 | awk '{print $1}')
E2E_PASSED=${E2E_PASSED:-0}
E2E_FAILED=${E2E_FAILED:-0}

# --- 2. Drive the infer path directly: record + replay a fixture
#         and assert cache-hit==1.0 with zero real executions.
INFER_OK=0
INFER_DETAIL=""
INFER_OUT=$("$PY" - <<'PY' 2>>"$LOG"
import json, os, tempfile, sys
from stepback import RecorderKey, record, replay, Executor
from tests.fixtures.agent import run_recorded_agent, LOOKUP_FIXED_ROW, fake_llm, fake_tool
from stepback.substitutions import ToolOutputSubstitution, SubstitutionSet

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "trace.sb")
key = RecorderKey.fresh()
with record(path, key=key) as rec:
    run_recorded_agent(rec)

# pure replay (no substitutions) -> all cache hits, no real exec
t = replay(path, hmac_key=key.hmac_key)
rep = t.replay_forward()
cache_hits = rep.cache_hit_count
real_execs = rep.real_executions
n = len(rep.steps)

# substitute path: replace tool output at step 2; expect propagation
subs = SubstitutionSet()
subs.add(ToolOutputSubstitution(at_step="step:2", fake_response=LOOKUP_FIXED_ROW))
t2 = replay(path, hmac_key=key.hmac_key)
rep2 = t2.run_replay(subs, Executor(llm=fake_llm, tool=fake_tool))
dirty = rep2.dirty_count

ok = (n > 0 and real_execs == 0 and cache_hits == n and dirty >= 1)
print(json.dumps({
    "ok": ok, "n_steps": n, "cache_hits": cache_hits,
    "real_execs": real_execs, "dirty_after_sub": dirty,
}))
PY
)
INFER_RC=$?
echo "infer_out=$INFER_OUT" >>"$LOG"

if [ "$INFER_RC" -eq 0 ] && [ -n "$INFER_OUT" ]; then
    case "$INFER_OUT" in
        *'"ok": true'*) INFER_OK=1 ;;
    esac
    INFER_DETAIL="$INFER_OUT"
else
    INFER_DETAIL="infer harness crashed (rc=$INFER_RC)"
fi

# --- 3. Score: 0.5 weight on pytest e2e, 0.5 weight on direct infer.
TOTAL_E2E=$((E2E_PASSED + E2E_FAILED))
if [ "$TOTAL_E2E" -gt 0 ]; then
    E2E_FRAC=$(awk -v p="$E2E_PASSED" -v t="$TOTAL_E2E" \
        'BEGIN{printf "%.4f", p/t}')
else
    E2E_FRAC=0.0
fi
SCORE=$(awk -v e="$E2E_FRAC" -v i="$INFER_OK" \
    'BEGIN{printf "%.4f", 0.5*e + 0.5*i}')

PASSED=false
awk -v s="$SCORE" 'BEGIN{exit !(s+0 >= 0.9)}' && PASSED=true

DETAILS="pytest_e2e: ${E2E_PASSED}/${TOTAL_E2E} passed (rc=${PYTEST_RC}); infer_direct: ok=${INFER_OK}; ${INFER_DETAIL}"

# escape details for JSON
DETAILS_ESC=$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$DETAILS")

cat > "$RESULT" <<EOF
{"score": ${SCORE}, "passed": ${PASSED}, "details": ${DETAILS_ESC}}
EOF

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HIST_ENTRY=$("$PY" -c '
import json, sys
print(json.dumps({
  "ts": sys.argv[1], "slot": "infer", "round": "r2e_gym",
  "score": float(sys.argv[2]), "passed": sys.argv[3] == "true",
  "pytest_passed": int(sys.argv[4]), "pytest_total": int(sys.argv[5]),
  "infer_ok": int(sys.argv[6]),
}))
' "$TS" "$SCORE" "$PASSED" "$E2E_PASSED" "$TOTAL_E2E" "$INFER_OK")
echo "$HIST_ENTRY" >> "$HIST"

# Final JSON line is the result.
cat "$RESULT"
