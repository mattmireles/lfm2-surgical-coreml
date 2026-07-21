# LFM2 Surgical Core ML — Agent Instructions

## Purpose

This repository measures whether LFM2.5-350M benefits from per-block-class
Core ML placement. Stage 0 proved that the short-convolution block admits to
ANE and the GQA block admits to GPU on both a Mac Studio and physical M2 iPad.
Stage 1 then failed gate G1a: 13 segment predictions consumed 37.43% of
segmented 512-token prefill. Do not run the cancelled device matrix or turn
the Stage 0 admission result into a placement-win claim.

## Working rules

- Simpler is better. Change the inference pipeline before changing the model.
- Use the real pinned `LiquidAI/LFM2.5-350M` checkpoint. Random weights are
  forbidden outside isolated unit fixtures that never produce scientific
  evidence.
- Exact enumerated shapes are frozen at 128, 256, 512, 1,024, and 2,048 prompt
  tokens. Range shapes do not satisfy the experiment.
- No dispatch table, no placement claim. A plan captured on one device says
  nothing about another OS build or device family.
- Keep fp16 as the only placement variable. Quantization and stateful models
  are follow-up experiments.
- Generated weights, packages, device plans, and measurements live under
  `outputs/` and remain uncommitted.
- Never commit `.env`, `.p8`, `.p12`, or provisioning-profile files.

## Documentation contract

Write for the next AI agent. Every function and property needs a formal
docstring. Document cross-file contracts and the reason behind every fixed
shape or numerical threshold. Replace magic numbers with named constants.

Start with:

- [Experiment specification](docs/experiment-spec-v1.1.md)
- [Stage 0 report](docs/stage0-report.md)
- [Stage 1 negative-result report](docs/stage1-report.md)
- [Public README](README.md)

Use the canonical external manuals rather than copying them:

- [LFM2 surgical prefill field guide](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md)
- [Core ML compute-unit scheduling](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [ANE transformer layout and compatibility](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/CoreML-ANE-transformer-layout-op-compatibility-guide.md)

## Verification ladder

1. `python -m py_compile` for changed Python files.
2. `pytest -q` for protocol and schema invariants.
3. Official Transformers parity for changed model equations.
4. Core ML numerical validation on all 32 prompts and five buckets.
5. Same-device compute-plan evidence for every placement claim.

The [Stage 0 report](docs/stage0-report.md#verification) contains the exact
commands and frozen hashes.

## License boundary

Repository code is MIT. Liquid AI weights and derived packages remain under
the LFM Open License v1.0. Never imply that MIT covers the model weights.
