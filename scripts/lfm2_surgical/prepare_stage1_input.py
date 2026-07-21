#!/usr/bin/env python3
"""Write the exact frozen 512-token G1a prompt IDs for the Swift harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lfm2_surgical.check_numerics import fixed_bucket_ids, load_tokenizer
from scripts.lfm2_surgical.checkpoint import checkpoint_identity
from scripts.lfm2_surgical.constants import (
    DEFAULT_CHECKPOINT_DIR,
    REAL_PROMPTS,
    validate_frozen_protocol,
)
from scripts.lfm2_surgical.export_segments import DECOMPOSITION_TAX_BUCKET

DEFAULT_OUTPUT = Path("outputs/lfm2_surgical/stage1/g1a_input_512.json")


def parse_args() -> argparse.Namespace:
    """Parse the frozen checkpoint and generated JSON output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Tokenize and repeat the first frozen real prompt to exactly 512 IDs."""

    args = parse_args()
    validate_frozen_protocol()
    tokenizer = load_tokenizer(args.checkpoint)
    token_ids = fixed_bucket_ids(
        tokenizer, REAL_PROMPTS[0], DECOMPOSITION_TAX_BUCKET
    )
    payload = {
        "bucket": DECOMPOSITION_TAX_BUCKET,
        "checkpoint": checkpoint_identity(args.checkpoint),
        "coremltools": ct.__version__,
        "prompt": REAL_PROMPTS[0],
        "token_ids": token_ids.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
