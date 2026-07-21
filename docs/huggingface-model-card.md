---
license: other
base_model: LiquidAI/LFM2.5-350M
pipeline_tag: text-generation
tags:
  - coreml
  - coremltools
  - apple-silicon
  - lfm2
  - negative-result
---

# LFM2.5-350M Surgical Core ML

This repository preserves the complete artifact line for a pre-registered
Core ML decomposition experiment on
[`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M).
The experiment produced a useful negative result: the operator classes admit
to different Apple accelerators, but separate Core ML predictions cost too
much for this finely interleaved architecture.

## Result

Stage 0 passed on physical hardware. The representative short-convolution
block placed 27/27 costed operations on the Neural Engine on an M2 iPad, while
the representative GQA block placed 84/84 operations on its GPU.

Stage 1 failed the frozen G1a stop gate:

| 512-token full-prefill control | Median | IQR | Same-host dispatch |
| --- | ---: | ---: | ---: |
| Monolithic | 36.032 ms | 0.352 ms | 797/797 GPU ops |
| 13 segments | 57.591 ms | 4.367 ms | 807/807 GPU ops |
| Added tax | **21.558 ms** | — | **37.43% of segmented prefill** |

The pre-registered maximum was 30%. The authoritative benchmark is a Release
Swift executable with direct fp16 `MLMultiArray` handoffs, three untimed
warmups per candidate, and 20 counterbalanced AB/BA measurements per
candidate. Paired final logits match exactly (`max_abs = 0.0`). The six-policy
and device matrices were cancelled as required by the frozen decision tree.
No iPhone latency, energy, decode, or thermal claim is made.

## Checkpoint identity

- Repository: `LiquidAI/LFM2.5-350M`
- Revision: `b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`
- `model.safetensors` SHA-256:
  `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`
- `config.json` SHA-256:
  `720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8`
- `tokenizer.json` SHA-256:
  `df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd`

## Repository layout

- `stage1/embedding.mlpackage`: enumerated-shape token embedding.
- `stage1/segments/*.mlpackage`: 13 maximal same-class layer-stack runs.
- `stage1/lm_head.mlpackage`: final normalization and tied projection.
- `stage1/monolithic_layer_stack_512.mlpackage`: fixed-512 G1a control.
- `stage1/*.json` and `stage1/dispatch_mac/`: export manifests, exact Swift
  timing JSON, and all 16 same-host compute-plan summaries.
- `source/`: conversion scripts, shared Swift runtime, tests, and frozen
  dependency files.
- `docs/`: experiment specification plus Stage 0 and Stage 1 reports.
- `LICENSE.lfm`: LFM Open License v1.0 governing the converted model
  artifacts.
- `LICENSE.code`: MIT license governing the original experiment code.

The 16 package-tree SHA-256 values are recorded in
`stage1/decomposition_tax_swift.json`. Core ML package UUIDs are regenerated
on conversion, so package-container hashes are evidence identities, not a
claim that independent conversions are byte-for-byte reproducible.

## Reproduce

The canonical source repository is
[`mattmireles/lfm2-surgical-coreml`](https://github.com/mattmireles/lfm2-surgical-coreml).
On Apple silicon with Python 3.11 and Xcode:

```bash
git clone https://github.com/mattmireles/lfm2-surgical-coreml.git
cd lfm2-surgical-coreml

python3.11 -m venv .venv
.venv/bin/pip install -r scripts/lfm2_surgical/requirements-stage0.txt

.venv/bin/python scripts/lfm2_surgical/extract_blocks.py
.venv/bin/python scripts/lfm2_surgical/export_blocks.py \
  --all-buckets --block both

.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope segments
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope embedding
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope head
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope monolith

.venv/bin/python scripts/lfm2_surgical/prepare_stage1_input.py
swift build -c release
.build/release/lfm2-surgical-benchmark \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --input-json outputs/lfm2_surgical/stage1/g1a_input_512.json \
  --warmups 3 --runs 20 \
  --out outputs/lfm2_surgical/stage1/decomposition_tax_swift.json
```

The benchmark exits nonzero after writing its JSON when G1a fails. Read
`docs/stage1-report.md` for the full protocol and commands that regenerate all
16 dispatch summaries.

## License

The converted Core ML packages are derivative model artifacts governed by the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE),
included here as `LICENSE.lfm`. The files were converted to Core ML and split
into class-contiguous segments; those modifications are described above and
in the export manifests. The license includes a commercial-use limitation for
legal entities at or above its stated revenue threshold.

Original conversion, validation, benchmark, and Swift orchestration code is
MIT-licensed separately under `LICENSE.code`.
