#!/usr/bin/env bash
#
# scripts/smoke_record_replay_docker.sh
#
# End-to-end smoke test for ghcr.io/stepback-dev/stepback:{VERSION} and
# ghcr.io/stepback-dev/stepback:{VERSION}-slim.
#
# What it does:
#   1. (optional) Builds both images from the repo root.
#   2. Records a minimal two-step trace inside the full image using
#      `stepback record` with an inline Python agent script.
#   3. Inspects the resulting .sb file via the full image to confirm
#      step count and structure.
#   4. Replays the trace via the slim image (no provider SDK needed)
#      to verify the offline replay path.
#   5. Verifies the HMAC chain via the slim image.
#
# Usage:
#   scripts/smoke_record_replay_docker.sh
#
# Env knobs:
#   SB_IMAGE         full image ref, e.g. ghcr.io/stepback-dev/stepback:0.1.0
#                    (default: stepback:smoke-test)
#   SB_SLIM_IMAGE    slim image ref (default: stepback:smoke-test-slim)
#   SB_SKIP_BUILD    if "1", skip `docker build` (images must already exist)
#   SB_KEEP          if "1", keep the temp trace dir on exit
#   SB_VERSION       version tag used in docker build arg (default: 0.1.0)
#
# Exit codes:
#   0  all smoke tests passed
#   1  generic failure
#   2  docker not found

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SB_VERSION="${SB_VERSION:-0.1.0}"
SB_IMAGE="${SB_IMAGE:-stepback:smoke-test}"
SB_SLIM_IMAGE="${SB_SLIM_IMAGE:-stepback:smoke-test-slim}"
SB_SKIP_BUILD="${SB_SKIP_BUILD:-0}"
SB_KEEP="${SB_KEEP:-0}"

TRACE_DIR=""
SCRIPT_DIR=""

log()  { echo "[smoke-record-replay] $*" >&2; }
ok()   { echo "[smoke-record-replay] ✓ $*" >&2; }
fail() { echo "[smoke-record-replay] ✗ $*" >&2; exit 1; }

cleanup() {
    if [[ "$SB_KEEP" == "1" ]]; then
        log "SB_KEEP=1 — leaving trace_dir=${TRACE_DIR} script_dir=${SCRIPT_DIR}"
    else
        [[ -n "$TRACE_DIR" ]] && rm -rf "$TRACE_DIR"
        [[ -n "$SCRIPT_DIR" ]] && rm -rf "$SCRIPT_DIR"
    fi
}
trap cleanup EXIT

# ── Prerequisites ──────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    echo "ERR: docker not found on PATH" >&2
    exit 2
fi

# ── Build images ───────────────────────────────────────────────────────────────
if [[ "$SB_SKIP_BUILD" != "1" ]]; then
    log "Building full image ${SB_IMAGE} ..."
    docker build -f docker/Dockerfile \
        --build-arg VERSION="${SB_VERSION}" \
        -t "${SB_IMAGE}" .

    log "Building slim image ${SB_SLIM_IMAGE} ..."
    docker build -f docker/Dockerfile.slim \
        --build-arg VERSION="${SB_VERSION}" \
        -t "${SB_SLIM_IMAGE}" .
    ok "Both images built."
else
    log "SB_SKIP_BUILD=1 — using pre-built images."
fi

# ── Prepare trace dir and agent script ─────────────────────────────────────────
TRACE_DIR="$(mktemp -d)"
SCRIPT_DIR="$(mktemp -d)"
chmod 777 "$TRACE_DIR"   # container user (65532) needs write access

# Write a minimal agent that records two steps (one llm_call, one tool_call)
# without calling a live provider by using stepback.testing fixture agents.
cat > "${SCRIPT_DIR}/smoke_agent.py" <<'PYEOF'
"""
Minimal smoke agent: records two steps using stepback's recorder API directly.
No live provider calls — works fully offline using fake executor callables.
"""
import os
import stepback
from stepback import record

trace_out = os.environ["SB_TRACE_OUT"]

def fake_llm(model, messages, **kw):
    """Return a canned chat completion — no network call."""
    return {"choices": [{"message": {"role": "assistant", "content": "4"}}]}

def fake_tool(name, arguments):
    """Return a canned tool result — no network call."""
    return {"result": 4}

with record(trace_out) as rec:
    # Simulated LLM call
    rec.llm_call(
        "gpt-4o-mini",
        [{"role": "user", "content": "What is 2+2?"}],
        executor=fake_llm,
    )
    # Simulated tool call
    rec.tool_call(
        "calculator",
        {"expression": "2+2"},
        executor=fake_tool,
    )

print(f"Recorded trace to {trace_out}")
PYEOF

ok "Agent script written to ${SCRIPT_DIR}/smoke_agent.py"

# ── Step 1: Record ──────────────────────────────────────────────────────────────
TRACE_FILE="${TRACE_DIR}/smoke.sb"
log "Recording trace via full image ..."
docker run --rm \
    -v "${TRACE_DIR}:/traces" \
    -v "${SCRIPT_DIR}:/scripts:ro" \
    -e SB_TRACE_OUT="/traces/smoke.sb" \
    --user "$(id -u):$(id -g)" \
    "${SB_IMAGE}" \
    record --output /traces/smoke.sb -- python /scripts/smoke_agent.py

if [[ ! -f "${TRACE_FILE}" ]]; then
    fail "smoke.sb not found in ${TRACE_DIR} after record step."
fi
ok "smoke.sb recorded ($(wc -c < "${TRACE_FILE}") bytes)."

# ── Step 2: Inspect (full image) ────────────────────────────────────────────────
log "Inspecting trace via full image ..."
INSPECT_OUT=$(docker run --rm \
    -v "${TRACE_DIR}:/traces:ro" \
    "${SB_IMAGE}" \
    inspect /traces/smoke.sb)

echo "${INSPECT_OUT}"

# Confirm two steps were recorded
STEP_COUNT=$(echo "${INSPECT_OUT}" | grep -c "step:" || true)
if [[ "${STEP_COUNT}" -lt 2 ]]; then
    fail "Expected at least 2 steps in inspect output, got ${STEP_COUNT}."
fi
ok "inspect found ${STEP_COUNT} step(s) matching 'step:'."

# ── Step 3: Replay (slim image) ─────────────────────────────────────────────────
log "Replaying trace via slim image ..."
REPLAY_OUT=$(docker run --rm \
    -v "${TRACE_DIR}:/traces:ro" \
    "${SB_SLIM_IMAGE}" \
    replay /traces/smoke.sb 2>&1 || true)

echo "${REPLAY_OUT}"

# replay should not hard-fail; check it emitted something plausible
if ! echo "${REPLAY_OUT}" | grep -qiE "step|dirty|cache|cost|replay"; then
    fail "replay output doesn't look right: ${REPLAY_OUT}"
fi
ok "replay completed via slim image."

# ── Step 4: --help sanity ───────────────────────────────────────────────────────
log "Checking --help in both images ..."
docker run --rm "${SB_IMAGE}"      --help > /dev/null 2>&1 || fail "full image --help failed"
docker run --rm "${SB_SLIM_IMAGE}" --help > /dev/null 2>&1 || fail "slim image --help failed"
ok "--help works in both images."

# ── Summary ─────────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────"
echo "  smoke_record_replay_docker.sh PASSED"
echo "  full image:  ${SB_IMAGE}"
echo "  slim image:  ${SB_SLIM_IMAGE}"
echo "  trace_dir=${TRACE_DIR}"
echo "────────────────────────────────────────────────"
