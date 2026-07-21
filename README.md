# LFM2.5 Surgical Core ML

**27/27 conv-block operations prefer the Neural Engine on an M2 iPad. 84/84
GQA-block operations prefer its GPU.** Same LFM2.5-350M checkpoint. Same five
prompt buckets. Different block classes, deliberately placed.

This repository tested one claim: a hybrid model may run better when its graph
is split at natural operator boundaries and each segment is sent to the device
that actually admits it. Stage 0 proved admission. Stage 1 killed the pipeline:
13 segment predictions consumed **37.43%** of segmented 512-token prefill.
No phone latency or energy claim is made.

This is the third case study for *Surgical Inference*. The first two public
artifact lines are [Kokoro Core ML](https://github.com/mattmireles/kokoro-coreml)
and [Magenta RealTime 2 for iPhone](https://github.com/mattmireles/magenta-realtime-2-iphone),
with the MRT2 packages mirrored on
[Hugging Face](https://huggingface.co/mattmireles/magenta-realtime-2-iphone).

## Stage 0 result

The frozen checkpoint is
[`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M) at
revision `b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`.

| Physical target | Conv block, CPU + ANE | GQA block, CPU + GPU |
| --- | ---: | ---: |
| Mac Studio M2 Ultra | 28/28 ANE | 84/84 GPU |
| iPad Pro M2, iPadOS 26.5 | 27/27 ANE | 84/84 GPU |

The result holds for the canonical five-shape packages and fixed diagnostic
twins at 128, 256, 512, 1,024, and 2,048 tokens. Every reported plan is bound
to its package-tree SHA-256. The Mac/iPad conv-count difference is one
`split` operation absent from the iPad's costed plan; all shared op counts
match.

Numerically, all 25 output-by-bucket cells pass the frozen `1e-2` max-absolute
error gate across 32 registered prompts. The official Transformers 5.5 oracle
matches the independent fp32 implementation exactly through layers 0–2.

Read the complete [Stage 0 report](docs/stage0-report.md) and the frozen
[experiment specification](docs/experiment-spec-v1.1.md).

## Stage 1 kill result

The registered G1a gate compared the same layer stack under one placement:
GPU permitted everywhere, one real 512-token prompt, warmed execution, and
load/compile time excluded.

| Control | Median | IQR | GPU dispatch |
| --- | ---: | ---: | ---: |
| Monolithic full prefill | 36.032 ms | 0.352 ms | 797/797 ops |
| 13-segment full prefill | 57.591 ms | 4.367 ms | 807/807 ops |

The release Swift harness hands fp16 `MLMultiArray` outputs directly between
models, and paired logits match exactly. The `21.558 ms` delta is therefore
boundary cost, not Python overhead, model drift, or different placement. It is
`37.43%` of segmented prefill, beyond the frozen `30%` stop line.

Read the complete [Stage 1 negative-result report](docs/stage1-report.md).

## Selective follow-up kill result

A separately preregistered follow-up tested whether three coarser
two-convolution ANE islands could avoid the 13-segment boundary tax. The
zero-boundary `.all` monolith was slower and numerically different, and all
three fixed-512 convolution pairs failed the `1e-2` GPU-versus-ANE numerical
gate. Pair timing, tail export, and device runs were cancelled before they
could produce a misleading speed result.

Read the complete
[selective-prefill negative-result report](docs/selective-split-report.md).

## A17 Pro parity follow-up

One final untimed phone probe asked whether the selective pair's ANE numerical
failure was confined to the M2 Ultra. It was not. On an iPhone 15 Pro Max with
iOS 27.0 beta (`24A5380h`), the fixed C6-7 twin placed 56/56 costed operations
on the A17 Pro Neural Engine, exceeded the unchanged `1e-2` correctness gate
in all six comparisons, and changed the final token `941 -> 509`. C0-1 and
C3-4 were cancelled by the preregistered stop rule. No phone performance
sample was collected.

Read the complete [A17 Pro parity report](docs/a17-parity-report.md).

## Reproduce the exports

Stage 0 requires Apple silicon, macOS, Python 3.11, and Xcode. The measured
environment used coremltools 8.3.0 and PyTorch 2.6.0.

```bash
git clone https://github.com/mattmireles/lfm2-surgical-coreml.git
cd lfm2-surgical-coreml

python3.11 -m venv .venv
.venv/bin/pip install -r scripts/lfm2_surgical/requirements-stage0.txt

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LiquidAI/LFM2.5-350M",
    revision="b9d6e4e2d75f440b12a2b4d731c808004ecbbd89",
    local_dir="outputs/lfm2_surgical/hf_model",
)
PY

.venv/bin/python scripts/lfm2_surgical/extract_blocks.py
.venv/bin/python scripts/lfm2_surgical/export_blocks.py \
  --all-buckets --block both
.venv/bin/python scripts/lfm2_surgical/check_numerics.py --prompts 32
```

The canonical packages land at:

- `outputs/lfm2_surgical/stage0/conv_block.mlpackage`
- `outputs/lfm2_surgical/stage0/gqa_block.mlpackage`

Generate the Mac compute plans:

```bash
.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage0/conv_block.mlpackage \
  --compute-units CPU_AND_NE \
  --out outputs/lfm2_surgical/stage0/dispatch_mac_conv_enumerated.json

.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage0/gqa_block.mlpackage \
  --compute-units CPU_AND_GPU \
  --out outputs/lfm2_surgical/stage0/dispatch_mac_gqa_enumerated.json
```

Physical-device plans additionally require a registered development device,
Developer Mode, a mounted developer disk image, and an installed provisioning
profile. See the exact iPad commands and provenance fields in the
[Stage 0 report](docs/stage0-report.md#verification).

## Verify the official model oracle

Transformers 5.5 lives in a separate environment so the frozen Core ML export
environment does not drift.

```bash
python3.11 -m venv .venv-hf
.venv-hf/bin/pip install \
  -r scripts/lfm2_surgical/requirements-hf-reference.txt
.venv-hf/bin/python scripts/lfm2_surgical/check_hf_reference.py
```

## Why split the model?

LFM2.5-350M interleaves 10 short-convolution blocks and 6 grouped-query
attention blocks:

```text
C C A C C A C C A C A C A C A C
```

The conv and attention blocks expose different scheduling behavior. A single
monolithic `computeUnits` policy cannot express “conv to ANE, attention to
GPU.” Stage 1 exported the 13 contiguous operator-class runs and measured
exactly that cost. The gate failed, so the device matrix was cancelled.

## Technical references

The canonical reference manuals stay in `kokoro-coreml`; this repository links
to them instead of maintaining stale copies:

- [LFM2 surgical prefill Core ML field guide](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md)
- [Core ML compute-unit scheduling](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [ANE transformer layout and op compatibility](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/CoreML-ANE-transformer-layout-op-compatibility-guide.md)
- [iPhone Core ML device-lab runbook](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md)

## License

The code in this repository is MIT-licensed; see [LICENSE](LICENSE).

Liquid AI's model weights and derived model artifacts are **not** covered by
the MIT license. They remain governed by the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).
No model weights are committed here.

Stage 0 answered admission. Stage 1 answered the harder question: these
boundaries are not worth it at 350M scale with separate Core ML predictions.
