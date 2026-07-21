#!/usr/bin/env python3
"""Materialize the three plan-011 conv pairs at the registered bucket 512."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dump_device_compute_plan import package_tree_sha256

FIXED_BUCKET = 512
HIDDEN_SIZE = 1_024
CONV_STATE_WIDTH = 2


@dataclass(frozen=True)
class PairMaterialization:
    """One frozen conv-pair package and its explicit state inputs."""

    name: str
    state_inputs: tuple[str, str]


PAIR_MATERIALIZATIONS = (
    PairMaterialization(
        "segment_00_conv_0_1", ("conv_state_0_in", "conv_state_1_in")
    ),
    PairMaterialization(
        "segment_02_conv_3_4", ("conv_state_3_in", "conv_state_4_in")
    ),
    PairMaterialization(
        "segment_04_conv_6_7", ("conv_state_6_in", "conv_state_7_in")
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the frozen Stage 1 source and selective output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("outputs/lfm2_surgical/stage1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/lfm2_surgical/selective/fixed_pairs_512"),
    )
    return parser.parse_args()


def fixed_shape_map(spec: PairMaterialization) -> dict[str, tuple[int, ...]]:
    """Return every input shape so materialization cannot infer a wrong bucket."""

    shapes = {"hidden_states": (1, HIDDEN_SIZE, 1, FIXED_BUCKET)}
    shapes.update(
        {name: (1, HIDDEN_SIZE, CONV_STATE_WIDTH) for name in spec.state_inputs}
    )
    return shapes


def static_input_shapes(model: ct.models.MLModel) -> dict[str, tuple[int, ...]]:
    """Read fixed tensor shapes and reject any surviving shape flexibility."""

    shapes: dict[str, tuple[int, ...]] = {}
    for feature in model.get_spec().description.input:
        array = feature.type.multiArrayType
        if array.HasField("enumeratedShapes") or array.HasField("shapeRange"):
            raise RuntimeError(f"{feature.name} remained shape-flexible")
        shapes[feature.name] = tuple(array.shape)
    return shapes


def materialize_pair(
    stage1_directory: Path,
    output_directory: Path,
    spec: PairMaterialization,
) -> dict[str, object]:
    """Materialize one exact source package and return hash-bound provenance."""

    source = stage1_directory / "segments" / f"{spec.name}.mlpackage"
    destination = output_directory / f"{spec.name}.mlpackage"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        shutil.rmtree(destination)
    output_directory.mkdir(parents=True, exist_ok=True)

    source_model = ct.models.MLModel(str(source), skip_model_load=True)
    ct.models.utils.materialize_dynamic_shape_mlmodel(
        dynamic_shape_mlmodel=source_model,
        function_name_to_materialization_map={"main": fixed_shape_map(spec)},
        destination_path=str(destination),
    )
    fixed_model = ct.models.MLModel(str(destination), skip_model_load=True)
    observed_shapes = static_input_shapes(fixed_model)
    expected_shapes = fixed_shape_map(spec)
    if observed_shapes != expected_shapes:
        raise RuntimeError(
            f"{spec.name} fixed input shapes drifted: {observed_shapes}"
        )
    source_outputs = [
        feature.name for feature in source_model.get_spec().description.output
    ]
    fixed_outputs = [
        feature.name for feature in fixed_model.get_spec().description.output
    ]
    if fixed_outputs != source_outputs:
        raise RuntimeError(f"{spec.name} output contract drifted")
    return {
        "name": spec.name,
        "source_package": str(source),
        "source_sha256": package_tree_sha256(source),
        "fixed_package": str(destination),
        "fixed_sha256": package_tree_sha256(destination),
        "fixed_input_shapes": {
            name: list(shape) for name, shape in observed_shapes.items()
        },
        "outputs": fixed_outputs,
    }


def main() -> None:
    """Materialize only the registered three pairs and write one manifest."""

    args = parse_args()
    payload = {
        "schema_version": 1,
        "coremltools": ct.__version__,
        "fixed_bucket": FIXED_BUCKET,
        "method": "materialize_dynamic_shape_mlmodel",
        "pairs": [
            materialize_pair(args.stage1_dir, args.out_dir, spec)
            for spec in PAIR_MATERIALIZATIONS
        ],
    }
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
