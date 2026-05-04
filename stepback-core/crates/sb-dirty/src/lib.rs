//! Dirty-set propagation over an SB-Trace DAG.
//!
//! Given a recorded trace and one or more substitutions, compute the
//! set of downstream steps whose canonical inputs would re-hash
//! differently and therefore must be re-executed. Every other step
//! can be served from the per-step content-addressed cache.
//!
//! ```text
//! Given:
//!   trace T = [s0, s1, ..., sN]
//!   substitution sigma at step sk
//!
//! Compute dirty-set D:
//!   D := {sk}
//!   for i in (k+1)..=N:
//!     inputs'_i := propagate sigma and D into si.inputs
//!     if hash(inputs'_i) != si.inputs_hash:
//!       D := D ∪ {si}
//! ```
//!
//! The Python implementation in `stepback/divergence.py` is the
//! reference. This crate ports the *types* and the linear-trace
//! propagation kernel; the DAG generalisation, parallel-branch
//! handling, and distributed scheduling land in later steps when the
//! Rust port reaches feature parity.

use std::collections::{BTreeSet, HashMap};

use sb_canonical::hash_value;
use sb_format::RecordedStep;
use serde_json::Value;
use thiserror::Error;

/// One substitution: replace some logical input on the named step.
///
/// A substitution is *typed* by `kind` so a substitution-aware
/// re-hasher can know which slice of `step.inputs` to splice in. The
/// Python `substitutions.py` module defines the canonical kinds:
/// `prompt`, `tool_output`, `model_id`, `policy`, `temperature`, ...
#[derive(Debug, Clone)]
pub struct Substitution {
    pub step_id: String,
    pub kind: String,
    pub value: Value,
}

/// Errors raised while computing a dirty-set.
#[derive(Debug, Error)]
pub enum DirtyError {
    #[error("substitution targeted unknown step_id {0:?}")]
    UnknownStepId(String),
    #[error("could not canonicalise rewritten inputs: {0}")]
    Canonical(#[from] sb_canonical::CanonicalError),
}

/// Result of a dirty-set computation.
#[derive(Debug, Clone, Default)]
pub struct DirtySet {
    /// Step IDs that must be re-executed, in trace order.
    pub steps: Vec<String>,
    /// Convenience set for membership tests.
    pub steps_set: BTreeSet<String>,
}

impl DirtySet {
    /// True if `step_id` would need to be re-executed under the
    /// substitution that produced this dirty set.
    pub fn contains(&self, step_id: &str) -> bool {
        self.steps_set.contains(step_id)
    }

    /// Number of steps that need real LLM/tool execution. The cached
    /// steps are `trace.len() - dirty.len()`.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True iff no steps are dirty (e.g. the substitution was a no-op).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Linear-trace dirty-set kernel. Returns the set of step IDs whose
/// inputs would change under `subs`.
///
/// `rewrite_inputs` lets the caller plug in the substitution semantics
/// (which keys of `step.inputs` to splice). The default
/// `default_rewrite_inputs` treats `kind` as a top-level key on
/// `step.inputs` — the same shape the Python `propagate_substitution`
/// uses for unit tests, but real recorders will install richer
/// rewrites via this hook.
pub fn compute_dirty_set<F>(
    trace: &[RecordedStep],
    subs: &[Substitution],
    mut rewrite_inputs: F,
) -> Result<DirtySet, DirtyError>
where
    F: FnMut(&RecordedStep, &[&Substitution], &BTreeSet<String>) -> Value,
{
    let by_id: HashMap<&str, &RecordedStep> =
        trace.iter().map(|s| (s.step_id.as_str(), s)).collect();

    for s in subs {
        if !by_id.contains_key(s.step_id.as_str()) {
            return Err(DirtyError::UnknownStepId(s.step_id.clone()));
        }
    }

    let mut dirty: BTreeSet<String> = BTreeSet::new();
    let mut order: Vec<String> = Vec::new();

    for step in trace {
        let direct: Vec<&Substitution> =
            subs.iter().filter(|s| s.step_id == step.step_id).collect();

        let touched_by_dirty = step
            .parent_step_id
            .as_ref()
            .map(|p| dirty.contains(p))
            .unwrap_or(false);

        if direct.is_empty() && !touched_by_dirty && !any_input_references_dirty(step, &dirty) {
            continue;
        }

        let new_inputs = rewrite_inputs(step, &direct, &dirty);
        let new_hash = hash_value(&new_inputs)?;

        if new_hash != step.inputs_hash {
            dirty.insert(step.step_id.clone());
            order.push(step.step_id.clone());
        }
    }

    Ok(DirtySet {
        steps: order,
        steps_set: dirty,
    })
}

/// Default no-op rewrite: returns `step.inputs` as-is unless a
/// substitution's `kind` matches a top-level key. Useful for tests
/// and as a starting point for richer adapters.
pub fn default_rewrite_inputs(
    step: &RecordedStep,
    direct: &[&Substitution],
    _dirty: &BTreeSet<String>,
) -> Value {
    let mut out = step.inputs.clone();
    if let Value::Object(map) = &mut out {
        for sub in direct {
            map.insert(sub.kind.clone(), sub.value.clone());
        }
    }
    out
}

/// Heuristic: did any string field on `step.inputs` mention a dirty
/// step ID? This mirrors the conservative back-edge detection in the
/// Python kernel for the linear case. Real DAG support replaces this
/// with explicit input-provenance edges.
fn any_input_references_dirty(step: &RecordedStep, dirty: &BTreeSet<String>) -> bool {
    fn walk(v: &Value, dirty: &BTreeSet<String>) -> bool {
        match v {
            Value::String(s) => dirty.iter().any(|d| s.contains(d.as_str())),
            Value::Array(items) => items.iter().any(|x| walk(x, dirty)),
            Value::Object(map) => map.values().any(|x| walk(x, dirty)),
            _ => false,
        }
    }
    walk(&step.inputs, dirty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sb_format::StepKind;
    use serde_json::json;

    fn step(id: &str, parent: Option<&str>, inputs: Value) -> RecordedStep {
        let inputs_hash = hash_value(&inputs).unwrap();
        RecordedStep {
            step_id: id.into(),
            step_kind: StepKind::LlmCall,
            parent_step_id: parent.map(|s| s.into()),
            inputs,
            outputs: Value::Null,
            inputs_hash,
            nondeterminism_hash: None,
            wallclock_ns: None,
            cpu_ns: None,
            cost_usd: None,
        }
    }

    #[test]
    fn no_substitution_means_empty_dirty_set() {
        let trace = vec![
            step("a", None, json!({"prompt": "hi"})),
            step("b", Some("a"), json!({"prompt": "ho"})),
        ];
        let d = compute_dirty_set(&trace, &[], default_rewrite_inputs).unwrap();
        assert!(d.is_empty());
    }

    #[test]
    fn substitution_dirties_only_the_target_for_independent_steps() {
        let trace = vec![
            step("a", None, json!({"prompt": "hi"})),
            step("b", None, json!({"prompt": "ho"})),
        ];
        let subs = vec![Substitution {
            step_id: "a".into(),
            kind: "prompt".into(),
            value: json!("HEY"),
        }];
        let d = compute_dirty_set(&trace, &subs, default_rewrite_inputs).unwrap();
        assert_eq!(d.steps, vec!["a".to_string()]);
        assert!(d.contains("a"));
        assert!(!d.contains("b"));
    }

    #[test]
    fn substitution_propagates_through_parent_link() {
        let trace = vec![
            step("a", None, json!({"prompt": "hi"})),
            step("b", Some("a"), json!({"prompt": "ho"})),
        ];
        let subs = vec![Substitution {
            step_id: "a".into(),
            kind: "prompt".into(),
            value: json!("HEY"),
        }];
        let d = compute_dirty_set(&trace, &subs, |step, direct, _dirty| {
            let mut out = step.inputs.clone();
            if let Value::Object(map) = &mut out {
                for s in direct {
                    map.insert(s.kind.clone(), s.value.clone());
                }
                if step.parent_step_id.as_deref() == Some("a") {
                    // Pretend the child consumes the parent's prompt.
                    map.insert("from_parent".into(), json!("HEY"));
                }
            }
            out
        })
        .unwrap();
        assert!(d.contains("a"));
        assert!(d.contains("b"));
    }

    #[test]
    fn unknown_target_step_is_rejected() {
        let trace = vec![step("a", None, json!({}))];
        let subs = vec![Substitution {
            step_id: "ghost".into(),
            kind: "prompt".into(),
            value: json!("x"),
        }];
        let err = compute_dirty_set(&trace, &subs, default_rewrite_inputs).unwrap_err();
        matches!(err, DirtyError::UnknownStepId(_));
    }
}
