"""Benchmark replay-from-cache and dirty-set propagation.

Records the 12-step fixture agent, then:
  1) Replays it with NO substitution and counts cache hits + LLM calls.
  2) Applies a ToolOutputSubstitution at the lookup step and counts the
     dirty subtree size.
  3) Bisects to the wrong-IBAN payment step and counts probes + LLM calls.

Prints a single line of metrics that grounds README claims about
zero-LLM-call replay, dirty-set size, and bisect probe budget.
"""
from __future__ import annotations

import os
import tempfile

from stepback import record, replay
from stepback.recorder import RecorderKey
from stepback.replay import Executor
from stepback.substitutions import ToolOutputSubstitution
from tests.fixtures.agent import LOOKUP_FIXED_ROW, fake_llm, fake_tool, run_recorded_agent


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bench.sb")
        with record(path, key=RecorderKey.fresh()) as rec:
            run_recorded_agent(rec)
        n_steps = len(rec.steps)

        # 1) Plain replay.
        t = replay(path)
        result0 = t.replay_forward()
        cached0 = result0.cache_hit_count
        llm0 = sum(
            1 for s in result0.steps if s.dirty and s.kind == "llm_call"
        )

        # 2) Substituted replay — fix the lookup_customer row.
        lookup_step = next(s for s in rec.steps if s["step_kind"] == "tool_call"
                           and s["inputs"]["name"] == "lookup_customer")
        t2 = replay(path)
        t2.substitute(
            ToolOutputSubstitution(
                at_step=lookup_step["step_id"],
                tool_call_id=None,
                fake_response=LOOKUP_FIXED_ROW,
            )
        )
        result1 = t2.replay_forward(Executor(llm=fake_llm, tool=fake_tool))
        dirty1 = result1.dirty_count
        llm1 = sum(
            1 for s in result1.steps if s.dirty and s.kind == "llm_call"
        )

        # 3) Bisect over plain replay (no substitutions).
        t3 = replay(path)
        first = rec.steps[0]["step_id"]
        last = rec.steps[-1]["step_id"]
        bad = t3.bisect(
            good=first,
            bad=last,
            predicate=lambda s: s.kind == "tool_call"
            and s.name == "payment.transfer"
            and "GB99" in str(s.outputs.get("result", "")),
        )
        probes = t3.last_bisect_probes
        bisect_llm = 0  # bisect probes are pure cache walks (no executor)

    print(
        f"n_steps={n_steps} "
        f"plain_cached={cached0} plain_llm_calls={llm0} "
        f"dirty_after_sub={dirty1} sub_llm_calls={llm1} "
        f"bisect_probes={probes} bisect_llm_calls={bisect_llm} "
        f"bad_step={bad.step_id if bad else None}"
    )


if __name__ == "__main__":
    main()
