"""Protocol and schema tests for plan 010 Stage 0 and Stage 1 tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from coremltools.converters.mil.mil.scope import ScopeSource

from scripts.dump_device_compute_plan import (
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
from scripts.lfm2_surgical.segments import (
    CoreMLConvBlockWithState,
    ReferenceConvBlockWithState,
    build_segment_specs,
)


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


def test_explicit_profile_disables_automatic_portal_mutations() -> None:
    """Headless signing must not combine a fixed profile with portal updates."""

    profile_uuid = "11111111-2222-3333-4444-555555555555"
    assert signing_build_arguments(profile_uuid) == [
        "CODE_SIGN_STYLE=MANUAL",
        f"PROVISIONING_PROFILE={profile_uuid}"
    ]
    assert signing_build_arguments(None) == [
        "CODE_SIGN_STYLE=AUTOMATIC",
        "-allowProvisioningUpdates",
        "-allowProvisioningDeviceRegistration",
    ]


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
