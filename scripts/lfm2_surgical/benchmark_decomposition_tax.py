#!/usr/bin/env python3
"""Measure the frozen G1a monolithic-versus-segmented GPU prefill control."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfm2_surgical.blocks import causal_mask, rope_tables
from scripts.lfm2_surgical.check_numerics import (
    fixed_bucket_ids,
    load_tokenizer,
    package_tree_sha256,
)
from scripts.lfm2_surgical.checkpoint import (
    checkpoint_identity,
    load_config,
    load_embedding,
)
from scripts.lfm2_surgical.constants import (
    CAUSAL_MASK_NEGATIVE,
    DEFAULT_CHECKPOINT_DIR,
    REAL_PROMPTS,
    validate_frozen_protocol,
)
from scripts.lfm2_surgical.export_segments import DECOMPOSITION_TAX_BUCKET
from scripts.lfm2_surgical.segments import build_segment_specs

DEFAULT_STAGE1_DIR = Path("outputs/lfm2_surgical/stage1")
DEFAULT_WARMUPS = 3
DEFAULT_RUNS = 20
G1A_MAX_OVERHEAD_FRACTION = 0.30


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, Stage 1 artifact, warmup, and run-count options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    return parser.parse_args()


def median_iqr(values: list[float]) -> dict[str, float]:
    """Return the median and inclusive interquartile range in milliseconds."""

    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median_ms": statistics.median(values),
        "q1_ms": quartiles[0],
        "q3_ms": quartiles[2],
        "iqr_ms": quartiles[2] - quartiles[0],
    }


class DecompositionTaxHarness:
    """Loaded Core ML controls and immutable inputs for one G1a run."""

    def __init__(self, checkpoint_dir: Path, stage1_dir: Path) -> None:
        """Load exact packages with compilation outside the timed region."""

        self.config = load_config(checkpoint_dir)
        self.specs = build_segment_specs(self.config.layer_types)
        self.stage1_dir = stage1_dir
        tokenizer = load_tokenizer(checkpoint_dir)
        token_ids = fixed_bucket_ids(
            tokenizer, REAL_PROMPTS[0], DECOMPOSITION_TAX_BUCKET
        )
        self.hidden = (
            load_embedding(checkpoint_dir)
            .float()[token_ids]
            .unsqueeze(0)
            .to(torch.float16)
            .permute(0, 2, 1)
            .unsqueeze(2)
            .numpy()
        )
        cosine, sine = rope_tables(
            DECOMPOSITION_TAX_BUCKET,
            self.config.head_dim,
            self.config.rope_theta,
            torch.float32,
        )
        mask = causal_mask(
            DECOMPOSITION_TAX_BUCKET, CAUSAL_MASK_NEGATIVE, torch.float16
        )
        self.attention_inputs = {
            "cosine": cosine.numpy(),
            "sine": sine.numpy(),
            "attention_mask": mask.numpy(),
        }
        self.zero_states = {
            index: np.zeros(
                (1, self.config.hidden_size, self.config.conv_l_cache - 1),
                dtype=np.float16,
            )
            for index, layer_type in enumerate(self.config.layer_types)
            if layer_type == "conv"
        }
        compute_units = ct.ComputeUnit.CPU_AND_GPU
        self.monolith_path = (
            stage1_dir
            / f"monolithic_layer_stack_{DECOMPOSITION_TAX_BUCKET}.mlpackage"
        )
        self.monolith = ct.models.MLModel(
            str(self.monolith_path), compute_units=compute_units
        )
        self.segment_paths = [
            stage1_dir / "segments" / f"{spec.name}.mlpackage"
            for spec in self.specs
        ]
        self.segments = [
            ct.models.MLModel(str(path), compute_units=compute_units)
            for path in self.segment_paths
        ]

    def run_monolith(self) -> np.ndarray:
        """Run the fixed-512 no-boundary control once and return hidden output."""

        inputs = {"hidden_states": self.hidden, **self.attention_inputs}
        inputs.update(
            {
                f"conv_state_{index}_in": state
                for index, state in self.zero_states.items()
            }
        )
        return np.asarray(self.monolith.predict(inputs)["hidden_out"])

    def run_segmented(self) -> np.ndarray:
        """Run all thirteen GPU-permitted segments and return hidden output."""

        hidden = self.hidden
        for spec, model in zip(self.specs, self.segments, strict=True):
            inputs = {"hidden_states": hidden}
            if spec.layer_type == "conv":
                inputs.update(
                    {
                        f"conv_state_{index}_in": self.zero_states[index]
                        for index in spec.layer_indices
                    }
                )
            else:
                inputs.update(self.attention_inputs)
            output = model.predict(inputs)
            # Core ML's Python bridge materializes fp16 model outputs as fp32.
            # The explicit cast is the real boundary copy the split graph pays.
            hidden = np.asarray(output["hidden_out"], dtype=np.float16)
        return hidden


def timed_prediction(function) -> tuple[np.ndarray, float]:
    """Return one prediction output and elapsed warmed wall time in milliseconds."""

    start = time.perf_counter_ns()
    output = function()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return output, elapsed_ms


def run_benchmark(
    checkpoint_dir: Path, stage1_dir: Path, warmups: int, runs: int
) -> dict:
    """Run the counterbalanced G1a benchmark and evaluate the frozen gate."""

    if warmups < 1:
        raise ValueError("--warmups must be positive")
    if runs < 4:
        raise ValueError("--runs must be at least 4 for a stable IQR")
    harness = DecompositionTaxHarness(checkpoint_dir, stage1_dir)
    for _ in range(warmups):
        harness.run_monolith()
        harness.run_segmented()

    monolith_times: list[float] = []
    segmented_times: list[float] = []
    output_max_abs = 0.0
    run_order: list[str] = []
    for run_index in range(runs):
        order = ("monolithic", "segmented")
        if run_index % 2:
            order = tuple(reversed(order))
        run_order.append("->".join(order))
        outputs: dict[str, np.ndarray] = {}
        for candidate in order:
            if candidate == "monolithic":
                output, elapsed_ms = timed_prediction(harness.run_monolith)
                monolith_times.append(elapsed_ms)
            else:
                output, elapsed_ms = timed_prediction(harness.run_segmented)
                segmented_times.append(elapsed_ms)
            outputs[candidate] = output
        output_max_abs = max(
            output_max_abs,
            float(
                np.max(
                    np.abs(
                        outputs["monolithic"].astype(np.float32)
                        - outputs["segmented"].astype(np.float32)
                    )
                )
            ),
        )

    monolithic_summary = median_iqr(monolith_times)
    segmented_summary = median_iqr(segmented_times)
    overhead_ms = segmented_summary["median_ms"] - monolithic_summary["median_ms"]
    overhead_fraction_of_segmented = overhead_ms / segmented_summary["median_ms"]
    overhead_relative_to_monolith = overhead_ms / monolithic_summary["median_ms"]
    passed = overhead_fraction_of_segmented <= G1A_MAX_OVERHEAD_FRACTION
    return {
        "checkpoint": checkpoint_identity(checkpoint_dir),
        "bucket": DECOMPOSITION_TAX_BUCKET,
        "compute_units": "CPU_AND_GPU",
        "environment": {
            "host": platform.node(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "coremltools": ct.__version__,
        },
        "protocol": {
            "warmups_per_candidate": warmups,
            "measured_runs_per_candidate": runs,
            "run_order": run_order,
            "load_and_compile_excluded": True,
            "prompt": REAL_PROMPTS[0],
        },
        "monolithic": {
            **monolithic_summary,
            "times_ms": monolith_times,
            "package": str(harness.monolith_path),
            "package_sha256": package_tree_sha256(harness.monolith_path),
        },
        "segmented": {
            **segmented_summary,
            "times_ms": segmented_times,
            "segment_count": len(harness.segment_paths),
            "packages": [
                {"path": str(path), "sha256": package_tree_sha256(path)}
                for path in harness.segment_paths
            ],
        },
        "output_max_abs": output_max_abs,
        "overhead_ms": overhead_ms,
        "overhead_fraction_of_segmented": overhead_fraction_of_segmented,
        "overhead_relative_to_monolith": overhead_relative_to_monolith,
        "g1a_max_overhead_fraction": G1A_MAX_OVERHEAD_FRACTION,
        "passed": passed,
        "verdict": "GO" if passed else "KILL",
    }


def main() -> None:
    """Run G1a, save its durable JSON, and fail closed on a kill verdict."""

    args = parse_args()
    validate_frozen_protocol()
    report = run_benchmark(
        args.checkpoint, args.stage1_dir, args.warmups, args.runs
    )
    output_path = args.stage1_dir / "decomposition_tax.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    if not report["passed"]:
        raise SystemExit("G1a failed: segmentation overhead exceeds 30% of prefill")


if __name__ == "__main__":
    main()
