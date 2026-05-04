#!/usr/bin/env bash
#
# scripts/smoke_proxy_docker.sh — end-to-end smoke test for the
# `sb proxy` container images defined in docker/Dockerfile.{distroless,debug}.
#
# What it does:
#   1. (optional) builds the chosen image from the repo root.
#   2. starts the container with a host-mounted trace directory.
#   3. waits for /healthz, then drives a full StartTrace -> RecordStep x 3
#      -> EndTrace -> VerifyTrace cycle over plain HTTP using `curl` + python
#      (avoids a hard dependency on `jq`).
#   4. confirms the .sb file appears on the *host* filesystem and verifies
#      it with the in-tree Python reader (`stepback.trace_reader.verify_trace`)
#      so we exercise the cross-boundary byte round-trip end to end.
#
# Usage:
#   scripts/smoke_proxy_docker.sh [distroless|debug]
#
# Env knobs:
#   SB_PROXY_IMAGE       full image ref to use; if set, skips local build
#                        regardless of the positional arg.
#   SB_PROXY_SKIP_BUILD  if "1", skip `docker build` (image must already
#                        exist locally or pullable).
#   SB_PROXY_PORT        host port to publish (default: random ephemeral)
#   SB_PROXY_KEEP        if "1", do not stop+rm the container on exit.
#
# Exit codes:
#   0  smoke passed
#   1  generic failure
#   2  docker not available
#   3  proxy failed to become ready in time

set -euo pipefail

variant="${1:-distroless}"
case "$variant" in
    distroless|debug) ;;
    *)
        echo "ERR: unknown variant '$variant' (expected: distroless|debug)" >&2
        exit 1
        ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERR: docker not found on PATH" >&2
    exit 2
fi

image="${SB_PROXY_IMAGE:-stepback/proxy:${variant}}"
dockerfile="docker/Dockerfile.${variant}"

if [[ "${SB_PROXY_SKIP_BUILD:-0}" != "1" && -z "${SB_PROXY_IMAGE:-}" ]]; then
    echo "==> building $image from $dockerfile"
    docker build -f "$dockerfile" -t "$image" .
fi

# Pick a host port. If SB_PROXY_PORT is unset, ask the kernel for a free one.
host_port="${SB_PROXY_PORT:-}"
if [[ -z "$host_port" ]]; then
    host_port=$(python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
)
fi

trace_dir="$(mktemp -d -t sb-proxy-smoke-XXXXXX)"
chmod 0777 "$trace_dir"   # so the in-container nonroot user can write
container_name="sb-proxy-smoke-$$-${RANDOM}"

cleanup() {
    rc=$?
    if [[ "${SB_PROXY_KEEP:-0}" != "1" ]]; then
        docker rm -f "$container_name" >/dev/null 2>&1 || true
        rm -rf "$trace_dir"
    else
        echo "==> kept container=$container_name trace_dir=$trace_dir"
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

echo "==> starting $image as $container_name on host port $host_port"
docker run -d --rm --name "$container_name" \
    -p "127.0.0.1:${host_port}:4319" \
    -v "${trace_dir}:/var/lib/stepback/traces" \
    "$image" >/dev/null

base_url="http://127.0.0.1:${host_port}"

echo -n "==> waiting for /healthz"
ready=0
for _ in $(seq 1 60); do
    if curl -fsS -o /dev/null "${base_url}/healthz" 2>/dev/null; then
        ready=1
        echo " ... ok"
        break
    fi
    echo -n "."
    sleep 0.5
done
if [[ "$ready" -ne 1 ]]; then
    echo " ... never became ready" >&2
    docker logs "$container_name" >&2 || true
    exit 3
fi

# Drive a full record cycle. Keep the JSON wrangling in Python so we don't
# need jq inside the smoke test.
python3 - "$base_url" "$trace_dir" <<'PY'
import json, sys, urllib.request, hashlib, os

base_url, host_trace_dir = sys.argv[1], sys.argv[2]

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    r = urllib.request.Request(
        base_url + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(r, timeout=10) as resp:
        return resp.status, json.loads(resp.read() or b"{}")

def hash_obj(o):
    s = json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()

s, info = req("GET", "/v1/info")
assert s == 200, info
print(f"   proxy_version={info.get('proxy_version')} schema={info.get('schema')}")

s, start = req("POST", "/v1/traces", {"filename": "smoke.sb"})
assert s == 201, start
trace_id = start["trace_id"]
hmac_key = start["hmac_key_hex"]
trace_path = start["path"]
print(f"   started trace_id={trace_id} path={trace_path}")

for i in range(1, 4):
    inputs = {"prompt": f"hello {i}", "model": "gpt-4o-mini"}
    outputs = {"text": f"world {i}"}
    step = {
        "step_id": f"step:{i}",
        "step_kind": "llm_call",
        "name": "gpt-4o-mini",
        "parent_step_id": f"step:{i-1}" if i > 1 else None,
        "inputs": inputs,
        "outputs": outputs,
        "inputs_hash": hash_obj(inputs),
        "outputs_hash": hash_obj(outputs),
        "nondeterminism_hash": hash_obj({}),
        "wallclock_ns": 1_000_000 * i,
        "cost_usd": 0.0,
    }
    s, body = req("POST", f"/v1/traces/{trace_id}/steps", {"step": step})
    assert s == 200 and body["step_count"] == i, body

s, end = req("POST", f"/v1/traces/{trace_id}/end", {})
assert s == 200 and end["step_count"] == 3, end
print(f"   ended trace, step_count={end['step_count']}")

s, ver = req("POST", "/v1/verify", {"path": trace_path, "hmac_key_hex": hmac_key})
assert s == 200 and ver["ok"] and ver["step_count"] == 3, ver
print(f"   in-container verify ok ({ver['step_count']} steps)")

# The in-container path is /var/lib/stepback/traces/smoke.sb. Translate to
# the host bind-mount and verify the bytes from outside the container too.
host_path = os.path.join(host_trace_dir, "smoke.sb")
assert os.path.exists(host_path), host_path
size = os.path.getsize(host_path)
print(f"   host sees {host_path} ({size} bytes)")
assert size > 0

# Run the canonical Python reader against the host file. This is the real
# end-to-end test: bytes written inside a distroless container, surfaced via
# bind-mount, read by the canonical reader on the host.
sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
from stepback.trace_reader import verify_trace
v = verify_trace(host_path, bytes.fromhex(hmac_key))
assert len(v.steps) == 3, v.steps
print(f"   host-side verify_trace ok ({len(v.steps)} steps)")
print("PASS")
PY

echo "==> smoke test passed for image=$image"
