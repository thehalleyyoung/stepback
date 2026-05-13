# RAG pipeline

A retrieve → re-rank → synthesise pipeline that answers a company-profile
question by pulling passages from three topics, re-ranking the candidates,
and generating a graded answer.

## Agent flow (6 steps)

```
step 1  llm_call   decompose question → sub-queries
step 2  tool_call  retrieve(topic=history)
step 3  tool_call  retrieve(topic=products)
step 4  tool_call  retrieve(topic=leadership)
step 5  tool_call  rerank(passages=all, top_k=4)
step 6  llm_call   synthesise answer
step 7  tool_call  grade_answer
step 8  llm_call   refine or confirm
```

## Re-record

```bash
make -C examples rag_pipeline
```

## Replay with substitution

```python
from stepback import replay
from stepback.substitutions import ToolOutputSubstitution
from examples.rag_pipeline.agent import fake_llm, fake_tool

trace = replay("examples/rag_pipeline/trace.sb")

# Inject a different history passage.
retrieve_step = next(
    s for s in trace.recorded_steps
    if s.get("name") == "retrieve" and s.get("inputs", {}).get("topic") == "history"
)
trace.substitute(
    retrieve_step["step_id"],
    ToolOutputSubstitution(output={
        "passages": [{"id": "alt", "text": "Founded 1995 as spin-off.", "score": 0.88}],
        "topic": "history",
    }),
)

result = trace.replay_forward(
    llm_executors={"gpt-4o-2024-11-20": fake_llm},
    tool_executors={"retrieve": fake_tool, "rerank": fake_tool, "grade_answer": fake_tool},
)
dirty = [s for s in result.steps if s.get("dirty")]
print(f"dirty steps: {len(dirty)}")  # rerank, synth, grade, refine all dirty
```

## What to learn here

- Substituting a single retrieval result propagates dirtiness through the
  re-ranker, synthesiser, grader, and refiner.
- The two unaffected `retrieve` calls (products, leadership) stay cached.
