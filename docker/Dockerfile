# syntax=docker/dockerfile:1.7
#
# Official stepback image — full-featured CLI with provider shims.
#
# Suitable for:
#   * recording live agent runs (openai, anthropic, bedrock, gemini shims included)
#   * replaying, inspecting, verifying, bisecting, and minimizing recorded traces
#   * running the quickstart wizard
#
# NOT for production sidecar use — use docker/Dockerfile.distroless instead.
#
# Build:
#   docker build -f docker/Dockerfile \
#     --build-arg VERSION=$(python -c "import stepback; print(stepback.__version__)") \
#     -t ghcr.io/stepback-dev/stepback:0.1.0 .
#
# Run (record an agent):
#   docker run --rm \
#     -v "$PWD/traces:/traces" \
#     -v "$PWD/agent.py:/work/agent.py:ro" \
#     -e OPENAI_API_KEY="$OPENAI_API_KEY" \
#     ghcr.io/stepback-dev/stepback:0.1.0 \
#     record --output /traces/run.sb -- python /work/agent.py
#
# Run (inspect a recorded trace):
#   docker run --rm \
#     -v "$PWD/traces:/traces:ro" \
#     ghcr.io/stepback-dev/stepback:0.1.0 \
#     inspect /traces/run.sb

ARG PYTHON_VERSION=3.11
ARG VERSION=0.1.0

# ---------- Stage 1: builder ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /src

COPY pyproject.toml MANIFEST.in README.md LICENSE CITATION.cff ./
COPY stepback/ ./stepback/

RUN python -m pip install --upgrade pip build \
 && python -m build --wheel --outdir /wheels . \
 && python -m pip install --prefix=/install --no-compile \
        "/wheels/stepback-*.whl[shims,quickstart]"

# ---------- Stage 2: runtime ----------
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG VERSION=0.1.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root account matching the distroless nonroot UID/GID (65532).
RUN groupadd --system --gid 65532 stepback \
 && useradd  --system --uid 65532 --gid 65532 \
             --home /home/stepback --shell /bin/bash stepback \
 && mkdir -p /home/stepback /traces \
 && chown -R stepback:stepback /home/stepback /traces

# Copy the pre-built site-packages from the builder stage.
COPY --from=builder /install/lib/python3.11/site-packages/ \
     /usr/local/lib/python3.11/site-packages/
COPY --from=builder /install/bin/ /usr/local/bin/

VOLUME ["/traces"]

LABEL org.opencontainers.image.title="stepback" \
      org.opencontainers.image.description="Time-travel debugger for AI agents — record, replay, bisect, minimize." \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/stepback-dev/stepback" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.documentation="https://github.com/stepback-dev/stepback#readme"

USER stepback
WORKDIR /home/stepback

ENTRYPOINT ["stepback"]
CMD ["--help"]
