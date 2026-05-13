# Container images for `stepback`

Two sets of images are published under `ghcr.io/stepback-dev/`:

## `stepback` — general-purpose CLI (record, replay, inspect, …)

These images expose the full `stepback` CLI and are the right choice for
recording agent runs and analysing traces:

| Tag | Dockerfile | Base | Shims | When to use |
| --- | --- | --- | --- | --- |
| `ghcr.io/stepback-dev/stepback:<ver>` | `docker/Dockerfile` | `python:3.11-slim-bookworm` | openai, anthropic, bedrock, gemini, quickstart | Record live runs + full analysis |
| `ghcr.io/stepback-dev/stepback:<ver>-slim` | `docker/Dockerfile.slim` | `python:3.11-slim-bookworm` | none (base only) | Offline replay / inspect / verify |

Both run as UID 65532 (`nonroot`). Mount a host directory over `/traces` so
`.sb` files survive container exits.

## `stepback-proxy` — HTTP/gRPC recorder sidecar

These images run the `stepback proxy` subcommand and are used as sidecars
to record traces from non-Python runtimes:

| Tag | Base | Shell | Tools | When to use |
| --- | --- | --- | --- | --- |
| `stepback/proxy:distroless` | `gcr.io/distroless/python3-debian12:nonroot` | none | Python 3.11 + `cryptography` only | Production / sidecar |
| `stepback/proxy:debug`      | `python:3.11-slim-bookworm`                  | `bash` | `curl`, `jq`, `tini`, `procps`, gRPC extra | Local triage, CI smoke |

Both proxy images run as UID 65532 (`nonroot`) and listen on `0.0.0.0:4319`.
The trace directory is `/var/lib/stepback/traces`; mount a host directory over
it so `.sb` files survive container exits.

## Build

```bash
# from the repo root

# stepback CLI images
docker build -f docker/Dockerfile      -t ghcr.io/stepback-dev/stepback:0.1.0      .
docker build -f docker/Dockerfile.slim -t ghcr.io/stepback-dev/stepback:0.1.0-slim .

# Proxy images
docker build -f docker/Dockerfile.distroless -t stepback/proxy:distroless .
docker build -f docker/Dockerfile.debug      -t stepback/proxy:debug      .
```

## Record + Replay quick-start

```bash
mkdir -p ./traces

# 1. Record an agent run (full image — provider shims available)
docker run --rm \
  -v "$PWD/traces:/traces" \
  -v "$PWD/my_agent.py:/work/my_agent.py:ro" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/stepback-dev/stepback:0.1.0 \
  record --output /traces/run.sb -- python /work/my_agent.py

# 2. Inspect the recorded trace (slim image — no SDK needed)
docker run --rm \
  -v "$PWD/traces:/traces:ro" \
  ghcr.io/stepback-dev/stepback:0.1.0-slim \
  inspect /traces/run.sb

# 3. Replay with a substitution (slim image)
docker run --rm \
  -v "$PWD/traces:/traces:ro" \
  ghcr.io/stepback-dev/stepback:0.1.0-slim \
  replay /traces/run.sb --substitute "model@step:0=gpt-4o-mini"

# 4. Verify the HMAC chain (slim image)
docker run --rm \
  -v "$PWD/traces:/traces:ro" \
  -e SB_HMAC_KEY_HEX="$SB_HMAC_KEY_HEX" \
  ghcr.io/stepback-dev/stepback:0.1.0-slim \
  verify /traces/run.sb
```

## Run (proxy sidecar)

```bash
mkdir -p ./traces
docker run --rm -p 4319:4319 \
  -v "$PWD/traces:/var/lib/stepback/traces" \
  stepback/proxy:distroless

# In another shell:
curl -sf http://127.0.0.1:4319/healthz | jq
curl -sf http://127.0.0.1:4319/v1/info | jq
```

## Smoke tests

### stepback CLI images (record + replay)

The end-to-end smoke test for the CLI images is
`scripts/smoke_record_replay_docker.sh`. It builds both images, records a
minimal two-step trace inside the full image, inspects it, and replays it
with the slim image.

```bash
./scripts/smoke_record_replay_docker.sh
# Skip build (images must already exist):
SB_SKIP_BUILD=1 SB_IMAGE=ghcr.io/stepback-dev/stepback:0.1.0 \
  SB_SLIM_IMAGE=ghcr.io/stepback-dev/stepback:0.1.0-slim \
  ./scripts/smoke_record_replay_docker.sh
```

### Proxy images

The end-to-end smoke test is `scripts/smoke_proxy_docker.sh`. It builds the
image (default: distroless; pass `debug` as `$1` for the other), starts the
container with a bind-mounted host directory, drives a full `StartTrace →
RecordStep × 3 → EndTrace → VerifyTrace` cycle over HTTP, and then verifies
the resulting `.sb` file *from the host* using the in-tree Python reader to
prove the bytes round-trip across the container boundary.

```bash
./scripts/smoke_proxy_docker.sh                    # distroless image
./scripts/smoke_proxy_docker.sh debug              # debug image
SB_PROXY_IMAGE=ghcr.io/stepback-dev/proxy:distroless \
  SB_PROXY_SKIP_BUILD=1 ./scripts/smoke_proxy_docker.sh
```

The proxy smoke test is also wired into CI as the `docker-smoke` job in
`.github/workflows/docker.yml`, which builds both images on every PR that
touches `docker/`, `stepback/proxy/`, `pyproject.toml`, or the workflow
itself, and uploads the resulting `.sb` artifact for inspection.

## Image hygiene

* All images set OCI labels (`org.opencontainers.image.*`) for provenance and
  licence attribution.
* The distroless proxy image carries **no shell, package manager, or setuid
  binary**; `docker exec` will fail. That is by design — operators who need
  a shell pull the `:debug` tag instead.
* Neither proxy image bakes in the optional `[shims]` extra (provider SDKs).
  The proxy is recorder-agnostic — clients send fully-formed step dicts.
* The slim CLI image intentionally omits `[shims]`: it cannot record live
  agent runs but is ~3–4× smaller than the full image.
* All images run as non-root UID 65532 (`nonroot`), matching the distroless
  convention. When using bind mounts, ensure the host directory is writable
  by UID 65532 or pass `--user "$(id -u):$(id -g)"` to the container.
