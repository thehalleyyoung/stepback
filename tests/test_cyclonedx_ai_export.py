"""Tests for the enhanced CycloneDX-AI export (step 110).

Covers:
- Models: modelCard architecture, temperature, seed, token usage
- Prompts: system/user messages embedded as properties
- Tools: service components with schema, tool arguments on step component
- Datasets: dataset components when dataset_id is present
- Policy decisions: policy/safety nondeterminism sources surfaced
- Spec version 1.6, services + dependencies arrays
- Lossiness report mentions dropped fields
- Round-trip: output is valid JSON, all required BOM fields present
"""
from __future__ import annotations

import json
import os

import pytest

from stepback.exporters import export_cyclonedx_ai, ExportReport

# ---------------------------------------------------------------------- helpers


def _make_llm_step(
    step_id: str = "step:1",
    model: str = "gpt-4o",
    messages: list | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    tools: list | None = None,
    usage: dict | None = None,
    parent: str | None = None,
    dataset_id: str | None = None,
) -> dict:
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
    llm_req = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "tools": tools,
    }
    llm_resp: dict = {
        "choices": [
            {"finish_reason": "stop", "index": 0,
             "message": {"role": "assistant", "content": "Hi there!"}}
        ],
        "model": model,
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    step: dict = {
        "step_id": step_id,
        "step_kind": "llm_call",
        "name": model,
        "parent_step_id": parent,
        "inputs": llm_req,
        "outputs": llm_resp,
        "llm_request": llm_req,
        "llm_response": llm_resp,
    }
    if dataset_id:
        step["dataset_id"] = dataset_id
    return step


def _make_tool_step(
    step_id: str = "step:2",
    tool_name: str = "search_web",
    arguments: dict | None = None,
    parent: str = "step:1",
) -> dict:
    if arguments is None:
        arguments = {"query": "python testing"}
    return {
        "step_id": step_id,
        "step_kind": "tool_call",
        "name": tool_name,
        "parent_step_id": parent,
        "inputs": {"kind": "tool_call", "name": tool_name, "arguments": arguments},
        "outputs": {"result": {"hits": 3}},
    }


def _make_policy_step(
    step_id: str = "step:3",
    parent: str = "step:1",
    finish_reason: str = "content_filter",
) -> dict:
    step = _make_llm_step(step_id=step_id, parent=parent)
    step["nondeterminism"] = {
        "sources": [
            {"class": "safety", "provider": "openai", "triggered": True},
            {"class": "model_sampling", "seed": 42, "temperature": 0.0},
        ]
    }
    # override finish_reason in response
    step["llm_response"]["choices"][0]["finish_reason"] = finish_reason
    step["outputs"]["choices"][0]["finish_reason"] = finish_reason
    return step


# ---------------------------------------------------------------------- tests


class TestCycloneDXAIBasicStructure:
    def test_output_is_valid_json(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        assert isinstance(doc, dict)

    def test_bom_format_and_spec_version(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        assert doc["bomFormat"] == "CycloneDX"
        assert doc["specVersion"] == "1.6"

    def test_serial_number_urn_uuid(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        assert doc["serialNumber"].startswith("urn:uuid:")

    def test_metadata_timestamp_and_tools(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        meta = doc["metadata"]
        assert "timestamp" in meta
        assert any(t["name"] == "stepback" for t in meta["tools"])

    def test_returns_export_report(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        report = export_cyclonedx_ai(steps, out)
        assert isinstance(report, ExportReport)
        assert report.target_format == "cyclonedx_ai"
        assert report.step_count == 1

    def test_lossiness_dropped_fields(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        report = export_cyclonedx_ai(steps, out)
        dropped = " ".join(report.lossiness.dropped)
        assert "execution" in dropped or "sequence" in dropped
        assert "timing" in dropped or "wallclock" in dropped

    def test_lossiness_absent_replay(self, tmp_path):
        steps = [_make_llm_step()]
        out = str(tmp_path / "out.json")
        report = export_cyclonedx_ai(steps, out)
        absent = " ".join(report.lossiness.absent)
        assert "replay" in absent.lower()

    def test_empty_steps_produces_valid_bom(self, tmp_path):
        out = str(tmp_path / "out.json")
        report = export_cyclonedx_ai([], out)
        with open(out) as f:
            doc = json.load(f)
        assert doc["components"] == []
        assert report.step_count == 0


class TestCycloneDXAIModels:
    def test_llm_step_has_model_property(self, tmp_path):
        steps = [_make_llm_step(model="gpt-4o")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        comp = doc["components"][0]
        props = {p["name"]: p["value"] for p in comp["properties"]}
        assert props["gen_ai:model"] == "gpt-4o"

    def test_llm_step_has_model_card_architecture(self, tmp_path):
        steps = [_make_llm_step(model="claude-3-opus")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        comp = doc["components"][0]
        assert comp["modelCard"]["modelParameters"]["modelArchitecture"] == "claude-3-opus"

    def test_llm_step_temperature_in_properties(self, tmp_path):
        steps = [_make_llm_step(temperature=0.7)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert props["gen_ai:temperature"] == "0.7"

    def test_llm_step_seed_in_properties(self, tmp_path):
        steps = [_make_llm_step(seed=99)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert props["gen_ai:seed"] == "99"

    def test_llm_step_token_usage_in_properties(self, tmp_path):
        usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        steps = [_make_llm_step(usage=usage)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert props["gen_ai:token_usage.prompt_tokens"] == "20"
        assert props["gen_ai:token_usage.completion_tokens"] == "10"
        assert props["gen_ai:token_usage.total_tokens"] == "30"


class TestCycloneDXAIPrompts:
    def test_system_prompt_role_in_properties(self, tmp_path):
        messages = [
            {"role": "system", "content": "You are a payments agent."},
            {"role": "user", "content": "Pay invoice 123."},
        ]
        steps = [_make_llm_step(messages=messages)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert props["prompt:messages[0].role"] == "system"
        assert props["prompt:messages[1].role"] == "user"

    def test_prompt_content_in_properties(self, tmp_path):
        messages = [{"role": "user", "content": "Hello world!"}]
        steps = [_make_llm_step(messages=messages)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert props["prompt:messages[0].content"] == "Hello world!"

    def test_prompt_content_is_truncated(self, tmp_path):
        long_content = "x" * 10_000
        messages = [{"role": "user", "content": long_content}]
        steps = [_make_llm_step(messages=messages)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        value = props["prompt:messages[0].content"]
        assert len(value) <= 2048

    def test_multipart_content_is_joined(self, tmp_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ],
            }
        ]
        steps = [_make_llm_step(messages=messages)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert "Hello" in props.get("prompt:messages[0].content", "")


class TestCycloneDXAITools:
    def test_tool_step_creates_service_component(self, tmp_path):
        steps = [_make_llm_step(), _make_tool_step(tool_name="search_web")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        services = doc.get("services", [])
        names = [s["name"] for s in services]
        assert "search_web" in names

    def test_tool_service_has_correct_bom_ref(self, tmp_path):
        steps = [_make_llm_step(), _make_tool_step(tool_name="my_tool")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        services = {s["name"]: s for s in doc.get("services", [])}
        assert services["my_tool"]["bom-ref"] == "tool:my_tool"

    def test_tool_schema_from_llm_request_tools_array(self, tmp_path):
        tools_def = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_customer",
                    "description": "Looks up a customer by name.",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            }
        ]
        steps = [
            _make_llm_step(tools=tools_def),
            _make_tool_step(tool_name="lookup_customer"),
        ]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        services = {s["name"]: s for s in doc.get("services", [])}
        svc = services["lookup_customer"]
        assert svc["description"] == "Looks up a customer by name."
        # parameters schema must be present
        props = {p["name"]: p["value"] for p in svc["properties"]}
        assert "tool:parameters_schema" in props

    def test_tool_step_has_name_property(self, tmp_path):
        steps = [_make_llm_step(), _make_tool_step(tool_name="send_email")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        # find the tool_call component
        tool_comp = next(
            c for c in doc["components"] if c.get("bom-ref") == "step:2"
        )
        props = {p["name"]: p["value"] for p in tool_comp["properties"]}
        assert props["tool:name"] == "send_email"

    def test_tool_step_arguments_in_properties(self, tmp_path):
        args = {"query": "test query", "limit": 5}
        steps = [_make_llm_step(), _make_tool_step(arguments=args)]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        tool_comp = next(
            c for c in doc["components"] if c.get("bom-ref") == "step:2"
        )
        props = {p["name"]: p["value"] for p in tool_comp["properties"]}
        assert "tool:arguments" in props
        parsed = json.loads(props["tool:arguments"])
        assert parsed["query"] == "test query"

    def test_tool_dependency_links_step_to_service(self, tmp_path):
        steps = [_make_llm_step(), _make_tool_step(tool_name="search_web")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        deps = doc.get("dependencies", [])
        tool_deps = [d for d in deps if d["ref"] == "step:2"]
        assert len(tool_deps) >= 1
        assert "tool:search_web" in tool_deps[0]["dependsOn"]

    def test_no_duplicate_service_for_same_tool(self, tmp_path):
        steps = [
            _make_llm_step(),
            _make_tool_step(step_id="step:2", tool_name="ping"),
            _make_tool_step(step_id="step:3", tool_name="ping"),
        ]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        services = doc.get("services", [])
        ping_services = [s for s in services if s["name"] == "ping"]
        assert len(ping_services) == 1


class TestCycloneDXAIDatasets:
    def test_dataset_component_created_when_dataset_id_present(self, tmp_path):
        steps = [_make_llm_step(dataset_id="ds-evals-001")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        dataset_comps = [
            c for c in doc["components"] if c.get("bom-ref", "").startswith("dataset:")
        ]
        assert len(dataset_comps) == 1
        assert dataset_comps[0]["name"] == "ds-evals-001"

    def test_dataset_component_has_data_type(self, tmp_path):
        steps = [_make_llm_step(dataset_id="my-dataset")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        ds = next(
            c for c in doc["components"]
            if c.get("bom-ref") == "dataset:my-dataset"
        )
        assert ds["type"] == "data"

    def test_dataset_dependency_links_step_to_dataset(self, tmp_path):
        steps = [_make_llm_step(step_id="step:1", dataset_id="ds-001")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        deps = doc.get("dependencies", [])
        step_deps = [d for d in deps if d["ref"] == "step:1"]
        assert any("dataset:ds-001" in d["dependsOn"] for d in step_deps)

    def test_no_duplicate_dataset_for_same_id(self, tmp_path):
        steps = [
            _make_llm_step(step_id="step:1", dataset_id="shared-ds"),
            _make_llm_step(step_id="step:2", dataset_id="shared-ds"),
        ]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        ds_comps = [
            c for c in doc["components"] if c.get("bom-ref") == "dataset:shared-ds"
        ]
        assert len(ds_comps) == 1


class TestCycloneDXAIPolicyDecisions:
    def test_safety_source_surfaced_as_policy_property(self, tmp_path):
        steps = [_make_policy_step()]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert "policy:safety" in props
        parsed = json.loads(props["policy:safety"])
        assert parsed["class"] == "safety"

    def test_content_filter_finish_reason_surfaced(self, tmp_path):
        steps = [_make_policy_step(finish_reason="content_filter")]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert "policy:finish_reason" in props
        assert props["policy:finish_reason"] == "content_filter"

    def test_normal_stop_finish_reason_not_surfaced(self, tmp_path):
        steps = [_make_llm_step()]  # default finish_reason = "stop"
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert "policy:finish_reason" not in props

    def test_explicit_policy_decisions_field(self, tmp_path):
        step = _make_llm_step()
        step["policy_decisions"] = [{"action": "block", "rule": "pii-filter"}]
        out = str(tmp_path / "out.json")
        export_cyclonedx_ai([step], out)
        with open(out) as f:
            doc = json.load(f)
        props = {p["name"]: p["value"] for p in doc["components"][0]["properties"]}
        assert "policy:decisions" in props
        parsed = json.loads(props["policy:decisions"])
        assert parsed[0]["action"] == "block"


class TestCycloneDXAIWithFixtureAgent:
    """Integration test using the real 12-step fixture agent."""

    def test_fixture_agent_export_produces_valid_bom(self, tmp_path):
        from stepback.testing import run_recorded_agent
        from stepback.recorder import RecorderKey
        from stepback import record
        from stepback.trace_reader import verify_trace

        sb_path = str(tmp_path / "fixture.sb")
        key = RecorderKey.fresh()
        with record(sb_path, key=key) as rec:
            run_recorded_agent(rec)
        steps = verify_trace(sb_path, key.hmac_key).steps

        out = str(tmp_path / "bom.json")
        report = export_cyclonedx_ai(steps, out)

        with open(out) as f:
            doc = json.load(f)

        assert doc["bomFormat"] == "CycloneDX"
        assert doc["specVersion"] == "1.6"
        assert len(doc["components"]) == 12
        assert report.step_count == 12

    def test_fixture_agent_export_has_model_cards(self, tmp_path):
        from stepback.testing import run_recorded_agent
        from stepback.recorder import RecorderKey
        from stepback import record
        from stepback.trace_reader import verify_trace

        sb_path = str(tmp_path / "fixture.sb")
        key = RecorderKey.fresh()
        with record(sb_path, key=key) as rec:
            run_recorded_agent(rec)
        steps = verify_trace(sb_path, key.hmac_key).steps

        out = str(tmp_path / "bom.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)

        llm_comps = [
            c for c in doc["components"]
            if any(p["name"] == "stepback:step_kind" and p["value"] == "llm_call"
                   for p in c.get("properties", []))
        ]
        assert len(llm_comps) > 0
        for c in llm_comps:
            assert "modelCard" in c
            assert "modelParameters" in c["modelCard"]

    def test_fixture_agent_export_has_prompt_properties(self, tmp_path):
        from stepback.testing import run_recorded_agent
        from stepback.recorder import RecorderKey
        from stepback import record
        from stepback.trace_reader import verify_trace

        sb_path = str(tmp_path / "fixture.sb")
        key = RecorderKey.fresh()
        with record(sb_path, key=key) as rec:
            run_recorded_agent(rec)
        steps = verify_trace(sb_path, key.hmac_key).steps

        out = str(tmp_path / "bom.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)

        llm_comps = [
            c for c in doc["components"]
            if any(p["name"] == "stepback:step_kind" and p["value"] == "llm_call"
                   for p in c.get("properties", []))
        ]
        # at least first LLM step should have a system prompt property
        first_llm = llm_comps[0]
        prop_names = [p["name"] for p in first_llm["properties"]]
        assert any(n.startswith("prompt:messages[") for n in prop_names)

    def test_fixture_agent_export_has_tool_services(self, tmp_path):
        from stepback.testing import run_recorded_agent
        from stepback.recorder import RecorderKey
        from stepback import record
        from stepback.trace_reader import verify_trace

        sb_path = str(tmp_path / "fixture.sb")
        key = RecorderKey.fresh()
        with record(sb_path, key=key) as rec:
            run_recorded_agent(rec)
        steps = verify_trace(sb_path, key.hmac_key).steps

        out = str(tmp_path / "bom.json")
        export_cyclonedx_ai(steps, out)
        with open(out) as f:
            doc = json.load(f)

        # fixture agent has 6 tool_call steps → services should not be empty
        services = doc.get("services", [])
        assert len(services) > 0
        for svc in services:
            assert "name" in svc
            assert svc.get("bom-ref", "").startswith("tool:")
