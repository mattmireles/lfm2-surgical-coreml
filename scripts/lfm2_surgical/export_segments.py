#!/usr/bin/env python3
"""Export the complete segmented LFM2.5 prefill stack and GPU control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfm2_surgical.blocks import causal_mask, rope_tables
from scripts.lfm2_surgical.checkpoint import checkpoint_identity, load_config
from scripts.lfm2_surgical.constants import (
    CAUSAL_MASK_NEGATIVE,
    DEFAULT_CHECKPOINT_DIR,
    PREFILL_BUCKETS,
    validate_frozen_protocol,
)
from scripts.lfm2_surgical.export_blocks import (
    enumerated_counts,
    enumerated_shapes,
    gqa_mixed_precision_selector,
    package_size_bytes,
    replace_package,
)
from scripts.lfm2_surgical.segments import (
    CoreMLEmbedding,
    CoreMLLayerSegment,
    CoreMLLayerStack,
    CoreMLLMHead,
    SegmentSpec,
    build_segment_specs,
)

DEFAULT_STAGE1_DIR = Path("outputs/lfm2_surgical/stage1")
TRACE_BUCKET = PREFILL_BUCKETS[0]
DECOMPOSITION_TAX_BUCKET = 512


def parse_args() -> argparse.Namespace:
    """Parse the frozen checkpoint, output directory, and export scope."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument(
        "--scope",
        choices=("all", "segments", "embedding", "head", "monolith"),
        default="all",
    )
    parser.add_argument(
        "--all-buckets",
        action="store_true",
        help="affirm the frozen enumerated-shape protocol",
    )
    return parser.parse_args()


def example_common_inputs(config, sequence_length: int) -> tuple[torch.Tensor, ...]:
    """Create trace inputs shared by GQA segments and the monolithic control."""

    hidden = torch.zeros(
        1, config.hidden_size, 1, sequence_length, dtype=torch.float16
    )
    cosine, sine = rope_tables(
        sequence_length, config.head_dim, config.rope_theta, torch.float32
    )
    mask = causal_mask(sequence_length, CAUSAL_MASK_NEGATIVE, torch.float16)
    return hidden, cosine, sine, mask


def hidden_input(config, fixed_bucket: int | None = None) -> ct.TensorType:
    """Return the canonical ANE-layout hidden-state input description."""

    shape = (
        (1, config.hidden_size, 1, fixed_bucket)
        if fixed_bucket is not None
        else enumerated_shapes(lambda bucket: (1, config.hidden_size, 1, bucket))
    )
    return ct.TensorType(name="hidden_states", shape=shape, dtype=np.float16)


def attention_inputs(config, fixed_bucket: int | None = None) -> list[ct.TensorType]:
    """Return synchronized RoPE and causal-mask Core ML input descriptions."""

    cosine_shape = (
        (1, fixed_bucket, config.head_dim)
        if fixed_bucket is not None
        else enumerated_shapes(lambda bucket: (1, bucket, config.head_dim))
    )
    mask_shape = (
        (1, 1, fixed_bucket, fixed_bucket)
        if fixed_bucket is not None
        else enumerated_shapes(lambda bucket: (1, 1, bucket, bucket))
    )
    return [
        ct.TensorType(name="cosine", shape=cosine_shape, dtype=np.float32),
        ct.TensorType(name="sine", shape=cosine_shape, dtype=np.float32),
        ct.TensorType(name="attention_mask", shape=mask_shape, dtype=np.float16),
    ]


def state_input(config, layer_index: int) -> ct.TensorType:
    """Return one constant-size explicit convolution-state input description."""

    return ct.TensorType(
        name=f"conv_state_{layer_index}_in",
        shape=(1, config.hidden_size, config.conv_l_cache - 1),
        dtype=np.float16,
    )


def output_types_for_spec(spec: SegmentSpec) -> list[ct.TensorType]:
    """Return stable flat output names for one segment."""

    outputs = [ct.TensorType(name="hidden_out", dtype=np.float16)]
    for layer_index in spec.layer_indices:
        if spec.layer_type == "conv":
            outputs.append(
                ct.TensorType(name=f"conv_state_{layer_index}_out", dtype=np.float16)
            )
        else:
            outputs.extend(
                (
                    ct.TensorType(name=f"key_{layer_index}_out", dtype=np.float16),
                    ct.TensorType(name=f"value_{layer_index}_out", dtype=np.float16),
                )
            )
    return outputs


def convert_segment(config, checkpoint_dir: Path, out_dir: Path, spec: SegmentSpec) -> dict:
    """Trace and convert one class-contiguous segment over every frozen bucket."""

    module = CoreMLLayerSegment(config, checkpoint_dir, spec).eval()
    hidden, cosine, sine, mask = example_common_inputs(config, TRACE_BUCKET)
    if spec.layer_type == "conv":
        states = tuple(
            torch.zeros(
                1,
                config.hidden_size,
                config.conv_l_cache - 1,
                dtype=torch.float16,
            )
            for _ in spec.layer_indices
        )
        example_inputs = (hidden, *states)
        coreml_inputs = [hidden_input(config)] + [
            state_input(config, layer_index) for layer_index in spec.layer_indices
        ]
        precision = ct.precision.FLOAT16
    else:
        example_inputs = (hidden, cosine, sine, mask)
        coreml_inputs = [hidden_input(config), *attention_inputs(config)]
        precision = ct.transform.FP16ComputePrecision(
            op_selector=gqa_mixed_precision_selector
        )
    with torch.inference_mode():
        traced = torch.jit.trace(module, example_inputs, strict=True)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=coreml_inputs,
        outputs=output_types_for_spec(spec),
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    package = out_dir / "segments" / f"{spec.name}.mlpackage"
    replace_package(model, package)
    expected_flexible = {"hidden_states": len(PREFILL_BUCKETS)}
    if spec.layer_type == "full_attention":
        expected_flexible.update(
            {
                "cosine": len(PREFILL_BUCKETS),
                "sine": len(PREFILL_BUCKETS),
                "attention_mask": len(PREFILL_BUCKETS),
            }
        )
    observed_counts = {
        name: count for name, count in enumerated_counts(model).items() if count
    }
    if observed_counts != expected_flexible:
        raise RuntimeError(
            f"{spec.name} lost the frozen enumerated shapes: {observed_counts}"
        )
    return {
        "name": spec.name,
        "layer_type": spec.layer_type,
        "layer_indices": list(spec.layer_indices),
        "package": str(package),
        "bytes": package_size_bytes(package),
        "enumerated_input_counts": observed_counts,
        "state_inputs": [
            f"conv_state_{index}_in"
            for index in spec.layer_indices
            if spec.layer_type == "conv"
        ],
    }


def convert_embedding(config, checkpoint_dir: Path, out_dir: Path) -> dict:
    """Trace and convert the masked token embedding over the frozen buckets."""

    module = CoreMLEmbedding(checkpoint_dir).eval()
    token_ids = torch.zeros(1, TRACE_BUCKET, dtype=torch.int32)
    token_mask = torch.ones(1, 1, 1, TRACE_BUCKET, dtype=torch.float16)
    with torch.inference_mode():
        traced = torch.jit.trace(module, (token_ids, token_mask), strict=True)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="token_ids",
                shape=enumerated_shapes(lambda bucket: (1, bucket)),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="token_mask",
                shape=enumerated_shapes(lambda bucket: (1, 1, 1, bucket)),
                dtype=np.float16,
            ),
        ],
        outputs=[ct.TensorType(name="hidden_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    package = out_dir / "embedding.mlpackage"
    replace_package(model, package)
    counts = enumerated_counts(model)
    expected = {"token_ids": len(PREFILL_BUCKETS), "token_mask": len(PREFILL_BUCKETS)}
    if counts != expected:
        raise RuntimeError(f"embedding lost the frozen enumerated shapes: {counts}")
    return {"package": str(package), "bytes": package_size_bytes(package)}


def convert_head(config, checkpoint_dir: Path, out_dir: Path) -> dict:
    """Trace and convert the final norm plus tied one-position LM head."""

    module = CoreMLLMHead(config, checkpoint_dir).eval()
    hidden = torch.zeros(1, config.hidden_size, 1, 1, dtype=torch.float16)
    with torch.inference_mode():
        traced = torch.jit.trace(module, (hidden,), strict=True)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states",
                shape=(1, config.hidden_size, 1, 1),
                dtype=np.float16,
            )
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    package = out_dir / "lm_head.mlpackage"
    replace_package(model, package)
    counts = enumerated_counts(model)
    if counts != {"hidden_states": 0}:
        raise RuntimeError(f"LM head unexpectedly became shape-flexible: {counts}")
    return {"package": str(package), "bytes": package_size_bytes(package)}


def monolith_output_types(config) -> list[ct.TensorType]:
    """Return output names shared by the fixed GPU control and Python oracle."""

    outputs = [ct.TensorType(name="hidden_out", dtype=np.float16)]
    for layer_index, layer_type in enumerate(config.layer_types):
        if layer_type == "conv":
            outputs.append(
                ct.TensorType(name=f"conv_state_{layer_index}_out", dtype=np.float16)
            )
    for layer_index, layer_type in enumerate(config.layer_types):
        if layer_type == "full_attention":
            outputs.extend(
                (
                    ct.TensorType(name=f"key_{layer_index}_out", dtype=np.float16),
                    ct.TensorType(name=f"value_{layer_index}_out", dtype=np.float16),
                )
            )
    return outputs


def convert_monolith(config, checkpoint_dir: Path, out_dir: Path) -> dict:
    """Convert the fixed-512 no-boundary control used only by gate G1a."""

    module = CoreMLLayerStack(config, checkpoint_dir).eval()
    hidden, cosine, sine, mask = example_common_inputs(
        config, DECOMPOSITION_TAX_BUCKET
    )
    conv_indices = [
        index for index, layer_type in enumerate(config.layer_types) if layer_type == "conv"
    ]
    states = tuple(
        torch.zeros(
            1, config.hidden_size, config.conv_l_cache - 1, dtype=torch.float16
        )
        for _ in conv_indices
    )
    with torch.inference_mode():
        traced = torch.jit.trace(
            module, (hidden, cosine, sine, mask, *states), strict=True
        )
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            hidden_input(config, DECOMPOSITION_TAX_BUCKET),
            *attention_inputs(config, DECOMPOSITION_TAX_BUCKET),
            *[state_input(config, index) for index in conv_indices],
        ],
        outputs=monolith_output_types(config),
        compute_precision=ct.transform.FP16ComputePrecision(
            op_selector=gqa_mixed_precision_selector
        ),
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    package = out_dir / f"monolithic_layer_stack_{DECOMPOSITION_TAX_BUCKET}.mlpackage"
    replace_package(model, package)
    return {
        "package": str(package),
        "bytes": package_size_bytes(package),
        "fixed_bucket": DECOMPOSITION_TAX_BUCKET,
    }


def main() -> None:
    """Export the requested Stage 1 artifacts and write their durable manifest."""

    args = parse_args()
    validate_frozen_protocol()
    if not args.all_buckets:
        raise SystemExit("pass --all-buckets to affirm the frozen protocol")
    config = load_config(args.checkpoint)
    specs = build_segment_specs(config.layer_types)
    manifest: dict[str, object] = {
        "checkpoint": checkpoint_identity(args.checkpoint),
        "buckets": list(PREFILL_BUCKETS),
        "minimum_deployment_target": "iOS18",
        "coremltools": ct.__version__,
        "segments": [],
    }
    if args.scope in ("all", "segments"):
        manifest["segments"] = [
            convert_segment(config, args.checkpoint, args.out_dir, spec)
            for spec in specs
        ]
    if args.scope in ("all", "embedding"):
        manifest["embedding"] = convert_embedding(config, args.checkpoint, args.out_dir)
    if args.scope in ("all", "head"):
        manifest["lm_head"] = convert_head(config, args.checkpoint, args.out_dir)
    if args.scope in ("all", "monolith"):
        manifest["monolith"] = convert_monolith(config, args.checkpoint, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"export_{args.scope}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
