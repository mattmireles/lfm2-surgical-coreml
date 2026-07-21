"""Protocol and schema tests for plan 010 Stage 0 and Stage 1 tooling."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from coremltools.converters.mil.mil.scope import ScopeSource

import scripts.run_a17_parity as a17
from scripts.dump_device_compute_plan import (
    COMPUTE_UNITS,
    host_provenance,
    model_runner_working_directory,
    signing_build_arguments,
    validated_device_provenance,
)
from scripts.lfm2_surgical.checkpoint import (
    Lfm2Config,
    checkpoint_identity,
    effective_intermediate_size,
)
from scripts.lfm2_surgical.benchmark_decomposition_tax import median_iqr
from scripts.lfm2_surgical.constants import (
    CONV_LAYER_INDEX,
    EXPECTED_LAYER_TYPES,
    GQA_INPUT_LAYER_INDEX,
    GQA_LAYER_INDEX,
    NUMERICS_PROMPT_COUNT,
    PREFILL_BUCKETS,
    REAL_PROMPTS,
    validate_frozen_protocol,
)
from scripts.lfm2_surgical.export_blocks import gqa_mixed_precision_selector
from scripts.lfm2_surgical.materialize_fixed_pairs import (
    FIXED_BUCKET,
    PAIR_MATERIALIZATIONS,
    fixed_shape_map,
)
from scripts.lfm2_surgical.segments import (
    CoreMLConvBlockWithState,
    ReferenceConvBlockWithState,
    build_segment_specs,
)
from scripts.run_a17_parity import (
    COMPARISON_ORDERS,
    DEFAULT_BUNDLE_ID,
    DEFAULT_DEVICE_ID,
    DEFAULT_DEVICE_UDID,
    DEFAULT_PRODUCT_TYPE,
    DISPATCH_SCHEMA,
    EXPECTED_TOKEN,
    MAXIMUM_ABSOLUTE_TOLERANCE,
    PAIR_SPECS,
    REGISTERED_BUCKET,
    REGISTERED_PROMPT,
    authorized_next_pair,
    make_envelope,
    package_tree_sha256,
    validate_dispatch_payload,
    validate_envelope,
    validate_pair_result_payload,
)


def _a17_dispatch_payload(policy: str) -> dict:
    """Return one minimal valid same-phone C6-7 dispatch fixture."""

    pair = PAIR_SPECS[0]
    histogram = {"gpu": 58} if policy == "CPU_AND_GPU" else {"neuralEngine": 58}
    per_op_type = (
        {"ios18.conv": {"gpu": 12}}
        if policy == "CPU_AND_GPU"
        else {"ios18.conv": {"neuralEngine": 12}}
    )
    return {
        "compute_plan": {
            "per_op_type": per_op_type,
            "preferred_device_histogram": histogram,
            "provenance": {
                "compute_units": policy,
                "device_identifier": DEFAULT_DEVICE_ID,
                "device_os_build": "24A5380h",
                "device_product_type": DEFAULT_PRODUCT_TYPE,
                "device_udid": DEFAULT_DEVICE_UDID,
                "package_sha256": pair.fixed_sha256,
                "where": "device",
            },
            "total_costed_ops": 58,
        },
        "device": {
            "coredevice_identifier": DEFAULT_DEVICE_ID,
            "hardware_udid": DEFAULT_DEVICE_UDID,
            "os_build": "24A5380h",
            "product_type": DEFAULT_PRODUCT_TYPE,
        },
        "package_sha256": pair.fixed_sha256,
        "pair": pair.name,
        "policy": policy,
    }


def _a17_result_payload(error: float = MAXIMUM_ABSOLUTE_TOLERANCE) -> dict:
    """Return one complete synthetic result at a selected numerical boundary."""

    pair = PAIR_SPECS[0]
    passed = error <= MAXIMUM_ABSOLUTE_TOLERANCE
    rows = []
    for index, order in enumerate(COMPARISON_ORDERS):
        rows.append(
            {
                "ane_token": EXPECTED_TOKEN,
                "candidate_order": order,
                "gpu_ane_max_abs_by_output": dict.fromkeys(pair.outputs, error),
                "gpu_token": EXPECTED_TOKEN,
                "oracle_token": EXPECTED_TOKEN,
                "pair_output_max_abs": error,
                "passed": passed,
                "repetition": index,
                "repetition_kind": "first_use" if index == 0 else "warmed",
                "source_twin_max_abs": error,
                "source_twin_max_abs_by_output": dict.fromkeys(pair.outputs, error),
                "suffix_logit_max_abs": error,
            }
        )
    return {
        "app": {"bundle_identifier": DEFAULT_BUNDLE_ID},
        "artifacts": [
            {
                "name": f"{pair.name}_fixed512",
                "path": "fixed",
                "resource_name": f"{pair.name}_fixed512",
                "sha256": pair.fixed_sha256,
            }
        ],
        "bucket": REGISTERED_BUCKET,
        "captured_at_utc": "2026-07-21T00:00:00+00:00",
        "checkpoint": {
            "repository": "LiquidAI/LFM2.5-350M",
            "revision": "b9d6e4e2d75f440b12a2b4d731c808004ecbbd89",
        },
        "comparisons": rows,
        "device": {
            "coredevice_identifier": DEFAULT_DEVICE_ID,
            "hardware_udid": DEFAULT_DEVICE_UDID,
            "os_build": "24A5380h",
            "product_type": DEFAULT_PRODUCT_TYPE,
        },
        "dispatch": [
            {
                "compute_units": policy,
                "evidence_sha256": policy.lower(),
                "package_sha256": pair.fixed_sha256,
                "path": policy.lower(),
            }
            for policy in ("CPU_AND_GPU", "CPU_AND_NE")
        ],
        "maximum_absolute_tolerance": MAXIMUM_ABSOLUTE_TOLERANCE,
        "oracle": {"passed": True, "top_token": EXPECTED_TOKEN},
        "pair": pair.name,
        "performance_samples_recorded": 0,
        "precision": "fp16",
        "prompt": REGISTERED_PROMPT,
        "schema_version": 1,
        "verdict": "PAIR PASS" if passed else "KILL",
    }


def test_a17_registered_pair_order_is_falsification_first() -> None:
    """The M2 token-changing C6-7 pair must run before either cheaper failure."""

    assert tuple(pair.name for pair in PAIR_SPECS) == (
        "segment_04_conv_6_7",
        "segment_00_conv_0_1",
        "segment_02_conv_3_4",
    )


def test_a17_envelope_rejects_hash_mismatch() -> None:
    """A later edit must not silently change hash-bound JSON evidence."""

    document = make_envelope(DISPATCH_SCHEMA, _a17_dispatch_payload("CPU_AND_GPU"))
    assert validate_envelope(document, DISPATCH_SCHEMA)["policy"] == "CPU_AND_GPU"
    document["payload"]["policy"] = "CPU_AND_NE"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_envelope(document, DISPATCH_SCHEMA)


def test_a17_dispatch_gate_requires_distinct_gpu_and_ane_placement() -> None:
    """Policy labels alone must not admit a pair to the physical probe."""

    pair = PAIR_SPECS[0]
    assert validate_dispatch_payload(
        _a17_dispatch_payload("CPU_AND_GPU"), pair, "CPU_AND_GPU"
    )["passed"]
    assert validate_dispatch_payload(
        _a17_dispatch_payload("CPU_AND_NE"), pair, "CPU_AND_NE"
    )["passed"]

    wrong = _a17_dispatch_payload("CPU_AND_NE")
    wrong["compute_plan"]["per_op_type"]["ios18.conv"] = {"gpu": 1, "neuralEngine": 11}
    with pytest.raises(ValueError, match="not wholly ANE-preferred"):
        validate_dispatch_payload(wrong, pair, "CPU_AND_NE")


def test_a17_result_accepts_threshold_equality_and_six_rows() -> None:
    """The preregistered `<= 1e-2` boundary passes without an epsilon rewrite."""

    assert (
        validate_pair_result_payload(_a17_result_payload(), PAIR_SPECS[0])
        == "PAIR PASS"
    )


def test_a17_result_rejects_nonfinite_or_incomplete_outputs() -> None:
    """NaN, infinity, or a missing named state must fail closed."""

    nonfinite = _a17_result_payload()
    nonfinite["comparisons"][0]["gpu_ane_max_abs_by_output"]["hidden_out"] = math.nan
    with pytest.raises(ValueError, match="NaN, infinity"):
        validate_pair_result_payload(nonfinite, PAIR_SPECS[0])

    incomplete = _a17_result_payload()
    del incomplete["comparisons"][0]["source_twin_max_abs_by_output"]["conv_state_6_out"]
    with pytest.raises(ValueError, match="names drifted"):
        validate_pair_result_payload(incomplete, PAIR_SPECS[0])


def test_a17_result_rejects_row_token_and_dispatch_drift() -> None:
    """Row count, token identity, and dispatch identity are mandatory evidence."""

    short = _a17_result_payload()
    short["comparisons"].pop()
    with pytest.raises(ValueError, match="exactly six"):
        validate_pair_result_payload(short, PAIR_SPECS[0])

    token = _a17_result_payload()
    token["comparisons"][0]["ane_token"] = 509
    with pytest.raises(ValueError, match="pass flag drifted"):
        validate_pair_result_payload(token, PAIR_SPECS[0])

    dispatch = _a17_result_payload()
    dispatch["dispatch"][0]["package_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dispatch reference package"):
        validate_pair_result_payload(dispatch, PAIR_SPECS[0])


def test_a17_result_above_threshold_is_terminal_kill() -> None:
    """No majority vote or warmed-row averaging may rescue one failed row."""

    payload = _a17_result_payload(MAXIMUM_ABSOLUTE_TOLERANCE + 1.0e-6)
    assert validate_pair_result_payload(payload, PAIR_SPECS[0]) == "KILL"
    assert authorized_next_pair([payload]) is None


def test_a17_state_machine_authorizes_only_the_next_passing_pair() -> None:
    """Later pairs cannot run before every registered predecessor passes."""

    assert authorized_next_pair([]) == PAIR_SPECS[0].name
    first = _a17_result_payload()
    assert authorized_next_pair([first]) == PAIR_SPECS[1].name


def test_a17_dispatch_command_checks_the_prior_result_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch capture itself may not bypass the registered stop-first order."""

    def reject_prior_gate(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prior result gate checked")

    monkeypatch.setattr(a17, "_prior_pair_payloads", reject_prior_gate)
    arguments = SimpleNamespace(
        pair=PAIR_SPECS[1].name,
        provisioning_profile_uuid="profile",
        device_id=DEFAULT_DEVICE_ID,
        out_dir=Path("dispatch"),
    )
    with pytest.raises(RuntimeError, match="prior result gate checked"):
        a17.run_dispatch(arguments)


def test_a17_result_binds_build_revision_os_and_completion_time() -> None:
    """A stale phone payload must not be accepted as the current signed run."""

    payload = _a17_result_payload()
    payload["app"]["revision"] = "revision-a"
    assert (
        validate_pair_result_payload(
            payload,
            PAIR_SPECS[0],
            expected_app_revision="revision-a",
            expected_os_build="24A5380h",
            not_before_utc="2026-07-20T23:59:00+00:00",
        )
        == "PAIR PASS"
    )
    with pytest.raises(ValueError, match="revision drifted"):
        validate_pair_result_payload(
            payload, PAIR_SPECS[0], expected_app_revision="revision-b"
        )
    with pytest.raises(ValueError, match="OS build drifted"):
        validate_pair_result_payload(payload, PAIR_SPECS[0], expected_os_build="wrong")
    with pytest.raises(ValueError, match="predates"):
        validate_pair_result_payload(
            payload,
            PAIR_SPECS[0],
            not_before_utc="2026-07-21T00:01:00+00:00",
        )


def test_a17_package_tree_hash_binds_relative_names_and_bytes(tmp_path: Path) -> None:
    """Renaming or changing a staged package file must change its identity."""

    package = tmp_path / "probe.mlpackage"
    package.mkdir()
    (package / "Manifest.json").write_text("{}")
    first = package_tree_sha256(package)
    (package / "Manifest.json").write_text('{"changed":true}')
    assert package_tree_sha256(package) != first


def test_a17_result_schema_contains_no_performance_fields() -> None:
    """Correctness evidence must not acquire even one latency or duration field."""

    payload = _a17_result_payload()
    payload["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="prohibited performance field"):
        validate_pair_result_payload(payload, PAIR_SPECS[0])


def test_frozen_bucket_and_prompt_protocol() -> None:
    """The executable protocol must match the plan's preregistered constants."""

    validate_frozen_protocol()
    assert PREFILL_BUCKETS == (128, 256, 512, 1024, 2048)
    assert len(REAL_PROMPTS) == NUMERICS_PROMPT_COUNT == 32


def test_frozen_layer_selection_matches_real_350m_config() -> None:
    """The isolated layers must remain one conv and one full-attention block."""

    assert EXPECTED_LAYER_TYPES[CONV_LAYER_INDEX] == "conv"
    assert EXPECTED_LAYER_TYPES[GQA_INPUT_LAYER_INDEX] == "conv"
    assert EXPECTED_LAYER_TYPES[GQA_LAYER_INDEX] == "full_attention"
    assert EXPECTED_LAYER_TYPES == (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )


def test_effective_intermediate_size_matches_liquid_adjustment() -> None:
    """The 350M checkpoint's declared 6656 width must load as trained width 4608."""

    config = {
        "intermediate_size": 6656,
        "block_auto_adjust_ff_dim": True,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
    }
    assert effective_intermediate_size(config) == 4608


def test_effective_intermediate_size_can_disable_adjustment() -> None:
    """Explicit checkpoint opt-out must preserve the declared width."""

    config = {
        "intermediate_size": 6656,
        "block_auto_adjust_ff_dim": False,
    }
    assert effective_intermediate_size(config) == 6656


def test_checkpoint_identity_rejects_config_and_tokenizer_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights, graph config, and tokenizer must share one frozen identity."""

    import scripts.lfm2_surgical.checkpoint as checkpoint

    artifacts = {
        "model.safetensors": b"weights",
        "config.json": b'{"norm_eps": 1e-5}',
        "tokenizer.json": b'{"model": "frozen"}',
    }
    metadata_directory = tmp_path / ".cache/huggingface/download"
    metadata_directory.mkdir(parents=True)
    for filename, payload in artifacts.items():
        (tmp_path / filename).write_bytes(payload)
        (metadata_directory / f"{filename}.metadata").write_text(
            f"{checkpoint.CHECKPOINT_REVISION}\nblob\ntimestamp\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_MODEL_SHA256",
        hashlib.sha256(artifacts["model.safetensors"]).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_CONFIG_SHA256",
        hashlib.sha256(artifacts["config.json"]).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint,
        "CHECKPOINT_TOKENIZER_SHA256",
        hashlib.sha256(artifacts["tokenizer.json"]).hexdigest(),
    )
    checkpoint_identity.cache_clear()
    assert checkpoint_identity(tmp_path)["config_sha256"] == hashlib.sha256(
        artifacts["config.json"]
    ).hexdigest()

    (tmp_path / "config.json").write_bytes(b'{"norm_eps": 1e-6}')
    checkpoint_identity.cache_clear()
    with pytest.raises(ValueError, match="config.json digest changed"):
        checkpoint_identity(tmp_path)
    checkpoint_identity.cache_clear()


def test_protocol_rejects_accidental_prompt_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed prompt corpus must fail before any measurement is produced."""

    import scripts.lfm2_surgical.constants as constants

    monkeypatch.setattr(constants, "REAL_PROMPTS", constants.REAL_PROMPTS[:-1])
    with pytest.raises(ValueError, match="expected 32 real prompts"):
        constants.validate_frozen_protocol()


def test_gqa_fp32_selector_uses_named_module_scopes_only() -> None:
    """Precision must not leak from the approved norm/RoPE islands."""

    approved = SimpleNamespace(
        scopes={ScopeSource.TORCHSCRIPT_MODULE_NAME: ["attention_positioning", "42"]}
    )
    residual_add = SimpleNamespace(
        scopes={ScopeSource.TORCHSCRIPT_MODULE_NAME: ["207"]}
    )
    assert gqa_mixed_precision_selector(approved) is False
    assert gqa_mixed_precision_selector(residual_add) is True


def test_mobile_provenance_fails_closed_and_persists_exact_identity() -> None:
    """The stale-tunnel fallback must prove and retain the physical device."""

    device = SimpleNamespace(
        name="iDesk",
        identifier="core-device-id",
        udid="physical-udid",
        type=SimpleNamespace(value="ipad"),
    )
    details = {
        "identifier": "core-device-id",
        "deviceProperties": {
            "name": "iDesk",
            "osVersionNumber": "26.5",
            "osBuildUpdate": "23F77",
            "developerModeStatus": "enabled",
            "ddiServicesAvailable": True,
        },
        "hardwareProperties": {
            "deviceType": "iPad",
            "productType": "iPad14,3",
            "udid": "physical-udid",
        },
        "connectionProperties": {
            "pairingState": "paired",
            "transportType": "wired",
            "tunnelState": "connected",
        },
    }
    provenance = validated_device_provenance(device, details)
    assert provenance["device_udid"] == "physical-udid"
    assert provenance["device_os_build"] == "23F77"

    details["deviceProperties"]["ddiServicesAvailable"] = False
    with pytest.raises(RuntimeError, match="developer_disk_image"):
        validated_device_provenance(device, details)


def test_compute_plan_supports_scheduler_control_with_host_identity() -> None:
    """The `.all` control must retain enough provenance to bind a Mac row."""

    assert "ALL" in COMPUTE_UNITS
    provenance = host_provenance()
    assert provenance["where"] == "host"
    assert provenance["host"]
    assert provenance["operating_system"]
    assert provenance["macos_product_version"]
    assert provenance["macos_build"]


def test_explicit_profile_disables_automatic_portal_mutations() -> None:
    """Headless signing must not combine a fixed profile with portal updates."""

    profile_uuid = "11111111-2222-3333-4444-555555555555"
    identity = "A" * 40
    assert signing_build_arguments(profile_uuid, identity) == [
        "CODE_SIGN_STYLE=MANUAL",
        f"PROVISIONING_PROFILE={profile_uuid}",
        f"CODE_SIGN_IDENTITY={identity}",
    ]
    assert signing_build_arguments(None) == [
        "CODE_SIGN_STYLE=AUTOMATIC",
        "-allowProvisioningUpdates",
        "-allowProvisioningDeviceRegistration",
    ]
    with pytest.raises(ValueError, match="explicit profile"):
        signing_build_arguments(None, identity)


def test_runner_cache_isolated_by_complete_signing_identity() -> None:
    """A changed profile, team, or bundle must never reuse a stale signed app."""

    first = model_runner_working_directory("TEAMONE", "example.runner", "PROFILE-A")
    assert first == model_runner_working_directory(
        "TEAMONE", "example.runner", "PROFILE-A"
    )
    assert first != model_runner_working_directory(
        "TEAMONE", "example.runner", "PROFILE-B"
    )
    assert first != model_runner_working_directory(
        "TEAMTWO", "example.runner", "PROFILE-A"
    )
    assert first != model_runner_working_directory(
        "TEAMONE", "example.other", "PROFILE-A"
    )


def test_stage1_partition_is_thirteen_maximal_class_runs() -> None:
    """The frozen 16-layer order must become exactly thirteen segment predictions."""

    specs = build_segment_specs(EXPECTED_LAYER_TYPES)
    assert len(specs) == 13
    assert specs[0].name == "segment_00_conv_0_1"
    assert specs[1].name == "segment_01_gqa_2"
    assert specs[4].name == "segment_04_conv_6_7"
    assert specs[-1].name == "segment_12_conv_15"
    assert tuple(
        layer_index for spec in specs for layer_index in spec.layer_indices
    ) == tuple(range(16))


def test_selective_pair_materialization_is_exactly_the_frozen_three() -> None:
    """The evidence amendment must not become a partition or bucket search."""

    assert FIXED_BUCKET == 512
    assert tuple(spec.name for spec in PAIR_MATERIALIZATIONS) == (
        "segment_00_conv_0_1",
        "segment_02_conv_3_4",
        "segment_04_conv_6_7",
    )
    for spec in PAIR_MATERIALIZATIONS:
        shapes = fixed_shape_map(spec)
        assert shapes["hidden_states"] == (1, 1_024, 1, 512)
        assert set(shapes) == {"hidden_states", *spec.state_inputs}


def test_explicit_conv_state_matches_reference_equation() -> None:
    """The ANE-layout state handoff must preserve the fp32 causal equation."""

    torch.manual_seed(7)
    config = Lfm2Config(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        norm_eps=1.0e-5,
        conv_l_cache=3,
        conv_bias=False,
        rope_theta=10_000.0,
        vocab_size=16,
        layer_types=("conv",),
    )
    tensors = {
        "operator_norm.weight": torch.rand(4),
        "ffn_norm.weight": torch.rand(4),
        "feed_forward.w1.weight": torch.rand(8, 4) - 0.5,
        "feed_forward.w2.weight": torch.rand(4, 8) - 0.5,
        "feed_forward.w3.weight": torch.rand(8, 4) - 0.5,
        "conv.in_proj.weight": torch.rand(12, 4) - 0.5,
        "conv.conv.weight": torch.rand(4, 1, 3) - 0.5,
        "conv.out_proj.weight": torch.rand(4, 4) - 0.5,
    }
    reference = ReferenceConvBlockWithState(config, tensors).eval()
    candidate = CoreMLConvBlockWithState(config, tensors).eval()
    hidden = torch.rand(1, 6, 4) - 0.5
    state = torch.rand(1, 4, 2) - 0.5
    with torch.inference_mode():
        expected_hidden, expected_state = reference(hidden, state)
        actual_hidden, actual_state = candidate(
            hidden.to(torch.float16).transpose(1, 2).unsqueeze(2),
            state.to(torch.float16),
        )
    actual_hidden = actual_hidden[:, :, 0, :].transpose(1, 2).float()
    assert torch.max(torch.abs(expected_hidden - actual_hidden)).item() < 5.0e-3
    assert torch.max(torch.abs(expected_state - actual_state.float())).item() < 5.0e-3


def test_decomposition_tax_summary_uses_inclusive_iqr() -> None:
    """Durable G1a summaries must retain the registered median and IQR signals."""

    summary = median_iqr([10.0, 20.0, 30.0, 40.0])
    assert summary == {
        "median_ms": 25.0,
        "q1_ms": 17.5,
        "q3_ms": 32.5,
        "iqr_ms": 15.0,
    }
