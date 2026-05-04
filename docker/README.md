# Container images for `sb proxy`

Two flavours, both built from the same Python 3.11 source tree so a trace
recorded against one is byte-identical to the other:

| Tag | Base | Shell | Tools | When to use |
| --- | --- | --- | --- | --- |
| `stepback/proxy:distroless` | `gcr.io/distroless/python3-debian12:nonroot` | none | Python 3.11 + `cryptography` only | Production / sidecar |
| `stepback/proxy:debug`      | `python:3.11-slim-bookworm`                  | `bash` | `curl`, `jq`, `tini`, `procps`, gRPC extra | Local triage, CI smoke |

Both images run as UID 65532 (`nonroot`) and listen on `0.0.0.0:4319`. The
trace directory is `/var/lib/stepback/traces`; mount a host directory over
it so `.sb` files survive container exits.

## Build

```bash
# from the repo root
docker build -f docker/Dockerfile.distroless -t stepback/proxy:distroless .
docker build -f docker/Dockerfile.debug      -t stepback/proxy:debug      .
```

## Run

```bash
mkdir -p ./traces
docker run --rm -p 4319:4319 \
  -v "$PWD/traces:/var/lib/stepback/traces" \
  stepback/proxy:distroless

# In another shell:
curl -sf http://127.0.0.1:4319/healthz | jq
curl -sf http://127.0.0.1:4319/v1/info | jq
```

## Smoke test (record + verify into a mounted dir)

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

The smoke test is also wired into CI as the `docker-smoke` job in
`.github/workflows/docker.yml`, which builds both images on every PR that
touches `docker/`, `stepback/proxy/`, `pyproject.toml`, or the workflow
itself, and uploads the resulting `.sb` artifact for inspection.

## Image hygiene

* Both images set OCI labels (`org.opencontainers.image.*`) so registries
  can attribute provenance and licence.
* The distroless image carries **no shell, package manager, or setuid
  binary**; `docker exec` will fail. That is by design — operators who need
  a shell pull the `:debug` tag instead.
* Neither image bakes in the optional `[shims]` extra (provider SDKs). The
  proxy is recorder-agnostic — clients send fully-formed step dicts.
