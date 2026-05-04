//! Replay planner and execution traits for SB-Trace.
//!
//! Given a recorded trace and a dirty-set, this crate produces an
//! ordered plan that says, for each step, whether to serve it from
//! the per-step content-addressed cache or to re-execute it via a
//! caller-supplied [`StepExecutor`].
//!
//! The Python reference is `stepback/replay.py`. This crate owns the
//! *types and traits* that the eventual Rust executor and the WASM
//! verifier share; concrete distributed schedulers (Kafka workers,
//! object-storage cache) plug in via the [`StepCache`] trait.

use sb_dirty::DirtySet;
use sb_format::RecordedStep;
use serde_json::Value;
use thiserror::Error;

/// One entry in the replay plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanAction {
    /// Use the recorded outputs verbatim.
    Cached,
    /// Re-execute the step; the caller-supplied executor is invoked.
    Reexecute,
}

/// Ordered list of (step_id, action) pairs. Trace-order, not
/// execution-order — concurrent execution comes from a scheduler that
/// observes parent-child dependencies.
#[derive(Debug, Clone, Default)]
pub struct ReplayPlan {
    pub entries: Vec<(String, PlanAction)>,
}

impl ReplayPlan {
    /// Number of steps that will need real LLM/tool execution.
    pub fn dirty_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|(_, a)| *a == PlanAction::Reexecute)
            .count()
    }

    /// Number of steps that will be served from cache.
    pub fn cached_count(&self) -> usize {
        self.entries.len() - self.dirty_count()
    }
}

/// Build the replay plan for `trace` given a `dirty` set.
pub fn plan(trace: &[RecordedStep], dirty: &DirtySet) -> ReplayPlan {
    let entries = trace
        .iter()
        .map(|s| {
            let action = if dirty.contains(&s.step_id) {
                PlanAction::Reexecute
            } else {
                PlanAction::Cached
            };
            (s.step_id.clone(), action)
        })
        .collect();
    ReplayPlan { entries }
}

/// Errors during replay execution.
#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("step {step_id:?} required re-execution but no executor was registered")]
    MissingExecutor { step_id: String },
    #[error("executor failed on step {step_id:?}: {message}")]
    Executor { step_id: String, message: String },
    #[error("cache miss on step {step_id:?} that was marked Cached")]
    CacheMiss { step_id: String },
}

/// Pluggable per-step content-addressed cache. Backed by an in-memory
/// map for tests, by object storage in production.
pub trait StepCache {
    fn get(&self, step_id: &str) -> Option<Value>;
}

/// Pluggable executor for steps that the dirty-set marked dirty.
/// Returns the new outputs the step would produce under the
/// substitution that was applied.
pub trait StepExecutor {
    fn execute(&mut self, step: &RecordedStep) -> Result<Value, String>;
}

/// Drive a [`ReplayPlan`] to completion, returning the resulting
/// per-step outputs in trace order.
pub fn execute_plan(
    trace: &[RecordedStep],
    plan: &ReplayPlan,
    cache: &dyn StepCache,
    executor: &mut dyn StepExecutor,
) -> Result<Vec<(String, Value)>, ReplayError> {
    let mut out = Vec::with_capacity(plan.entries.len());
    for (step_id, action) in &plan.entries {
        let step = trace
            .iter()
            .find(|s| &s.step_id == step_id)
            .expect("plan entries are derived from trace");
        let value = match action {
            PlanAction::Cached => cache
                .get(step_id)
                .ok_or_else(|| ReplayError::CacheMiss { step_id: step_id.clone() })?,
            PlanAction::Reexecute => executor
                .execute(step)
                .map_err(|message| ReplayError::Executor {
                    step_id: step_id.clone(),
                    message,
                })?,
        };
        out.push((step_id.clone(), value));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sb_canonical::hash_value;
    use sb_format::StepKind;
    use std::collections::HashMap;

    fn s(id: &str, outputs: Value) -> RecordedStep {
        let inputs = serde_json::json!({"step_id": id});
        let inputs_hash = hash_value(&inputs).unwrap();
        RecordedStep {
            step_id: id.into(),
            step_kind: StepKind::LlmCall,
            parent_step_id: None,
            inputs,
            outputs,
            inputs_hash,
            nondeterminism_hash: None,
            wallclock_ns: None,
            cpu_ns: None,
            cost_usd: None,
        }
    }

    struct MapCache(HashMap<String, Value>);
    impl StepCache for MapCache {
        fn get(&self, step_id: &str) -> Option<Value> {
            self.0.get(step_id).cloned()
        }
    }

    struct CountingExec(usize);
    impl StepExecutor for CountingExec {
        fn execute(&mut self, step: &RecordedStep) -> Result<Value, String> {
            self.0 += 1;
            Ok(serde_json::json!({"reexecuted": step.step_id}))
        }
    }

    #[test]
    fn plan_marks_dirty_and_cached() {
        let trace = vec![s("a", Value::Null), s("b", Value::Null)];
        let mut dirty = DirtySet::default();
        dirty.steps.push("b".into());
        dirty.steps_set.insert("b".into());
        let p = plan(&trace, &dirty);
        assert_eq!(p.cached_count(), 1);
        assert_eq!(p.dirty_count(), 1);
        assert_eq!(p.entries[0], ("a".into(), PlanAction::Cached));
        assert_eq!(p.entries[1], ("b".into(), PlanAction::Reexecute));
    }

    #[test]
    fn execute_plan_uses_cache_for_clean_and_executor_for_dirty() {
        let trace = vec![s("a", serde_json::json!({"orig": "a"})), s("b", Value::Null)];
        let mut dirty = DirtySet::default();
        dirty.steps.push("b".into());
        dirty.steps_set.insert("b".into());
        let p = plan(&trace, &dirty);

        let mut cache = HashMap::new();
        cache.insert("a".to_string(), serde_json::json!({"orig": "a"}));
        let cache = MapCache(cache);
        let mut exec = CountingExec(0);

        let outs = execute_plan(&trace, &p, &cache, &mut exec).unwrap();
        assert_eq!(exec.0, 1, "only the dirty step is re-executed");
        assert_eq!(outs[0].1, serde_json::json!({"orig": "a"}));
        assert_eq!(outs[1].1, serde_json::json!({"reexecuted": "b"}));
    }

    #[test]
    fn cache_miss_on_clean_step_is_an_error() {
        let trace = vec![s("a", Value::Null)];
        let p = plan(&trace, &DirtySet::default());
        let cache = MapCache(HashMap::new());
        let mut exec = CountingExec(0);
        let err = execute_plan(&trace, &p, &cache, &mut exec).unwrap_err();
        matches!(err, ReplayError::CacheMiss { .. });
    }
}
