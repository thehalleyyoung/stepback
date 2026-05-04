#!/usr/bin/env bash
# Build wheel + sdist, install the wheel into a fresh venv, run smoke checks.
# Used both locally and (eventually) by CI to catch entry-point and packaging
# regressions described in 100_STEPS.md step 8.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rm -rf dist build stepback.egg-info

build_venv="$(mktemp -d)/build-venv"
python3 -m venv "$build_venv"
"$build_venv/bin/pip" install --quiet --upgrade pip build
"$build_venv/bin/python" -m build

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

python3 -m venv "$work_dir/venv"
"$work_dir/venv/bin/pip" install --quiet --upgrade pip
wheel_path="$(ls dist/stepback-*.whl | head -n1)"
"$work_dir/venv/bin/pip" install --quiet "$wheel_path"

echo "--- stepback --help ---"
"$work_dir/venv/bin/stepback" --help >/dev/null
echo "ok"

echo "--- import + version parity ---"
"$work_dir/venv/bin/python" - <<'PY'
import stepback
from importlib.metadata import version
assert stepback.__version__ == version("stepback"), (
    stepback.__version__,
    version("stepback"),
)
print("stepback", stepback.__version__)
PY

echo "--- py.typed shipped ---"
"$work_dir/venv/bin/python" - <<'PY'
import importlib.resources as ir
import stepback
files = ir.files(stepback)
assert (files / "py.typed").is_file(), "py.typed not shipped in wheel"
print("py.typed: ok")
PY

echo "--- one-step record + replay ---"
"$work_dir/venv/bin/python" - <<'PY'
import os, tempfile
from stepback import record, replay

with tempfile.TemporaryDirectory() as d:
    trace_path = os.path.join(d, "smoke.sb")

    def fake_llm(model, messages, **kwargs):
        return {
            "id": "smoke-1",
            "model": model,
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }

    with record(trace_path) as rec:
        rec.llm_call(
            "gpt-smoke",
            [{"role": "user", "content": "ping"}],
            executor=fake_llm,
        )

    t = replay(trace_path)
    steps = list(t.recorded_steps)
    assert len(steps) == 1, f"expected 1 step, got {len(steps)}"
    assert steps[0]["step_kind"] == "llm_call", steps[0]["step_kind"]
    print(f"recorded+replayed {len(steps)} step(s) from {trace_path}")
PY

echo "--- stepback inspect ---"
"$work_dir/venv/bin/python" - <<'PY'
import os, subprocess, tempfile, sys
from stepback import record

with tempfile.TemporaryDirectory() as d:
    trace_path = os.path.join(d, "inspect.sb")

    def fake_llm(model, messages, **kwargs):
        return {
            "id": "i-1", "model": model,
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }
    with record(trace_path) as rec:
        rec.llm_call("gpt-smoke",
                     [{"role": "user", "content": "ping"}],
                     executor=fake_llm)

    venv_bin = os.path.dirname(sys.executable)
    out = subprocess.run(
        [os.path.join(venv_bin, "stepback"), "inspect", trace_path,
         "--json"],
        capture_output=True, text=True, check=True,
    )
    assert "llm_call" in out.stdout, out.stdout[:200]
    print("inspect: ok")
PY

echo "smoke OK"
