# Selective Prefill Report: ANE Numerics Kill the Six-Piece Path

**Date:** 2026-07-20

**Status:** Terminal `KILL` at preregistered gate G0d

**Checkpoint:** `LiquidAI/LFM2.5-350M` at
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`

**Host:** Mac Studio M2 Ultra, macOS 26.5.2 (`25F84`)

**Runtime:** Release Swift, direct fp16 `MLMultiArray`, coremltools 8.3.0

## Decision

Do not export or test the proposed six-piece partition. Core ML admitted every
registered fixed-512 convolution pair entirely to the Neural Engine, but the
ANE-permitted executions failed the frozen `1e-2` numerical tolerance. One
pair also changed the final top token. Correctness failed before pair timing,
so aggregate pair economics, the layers-8-15 tail export, iPad runs, and iPhone
runs were cancelled.

This does not reverse the earlier 13-segment result. Plan 010 rejected a
fine-grained partition because 12 public boundaries consumed 37.43% of total
segmented prefill. This independently registered follow-up rejected a
lower-boundary partition earlier, because its proposed ANE islands were not
numerically interchangeable with GPU execution.

## Frozen protocol

The registered prompt was:

> Explain why fixed tensor shapes can help a compiler target an accelerator.

All gates used bucket 512, fp16 weights and tensors, three untimed warmups, 20
counterbalanced Release Swift measurements where timing was allowed, real
layer-entry activations, zero prefill states, package-tree SHA-256 identities,
and compute plans captured on the same host and OS build. The result binds all
19 packages used by G0a or the correctness oracle.

The timed run passed its durable host-noise preflight immediately before
warmup: one-minute load `4.7085` across 24 logical processors (`0.1962` per
processor), 59% system memory free, no other process at or above 50% CPU, and
nominal thermal state. Swap usage (`1570.75 MiB`) was recorded but was not a
kill threshold.

The public `MLComputePlan` API cannot select a non-default specialization from
an `EnumeratedShapes` package. Before pair evidence, the experiment therefore
materialized exactly the three preregistered bucket-512 twins. Each new twin
was hash-bound and checked against its enumerated source on the real input.
No layer equation, partition, compute policy, prompt, threshold, or precision
changed.

## Gate results

| Gate | Result | Evidence |
| --- | --- | --- |
| G0a: zero-boundary scheduler | Continue | `.all` used heterogeneous placement, but was slower and numerically different. |
| G0b: target admission | Pass | Each ANE-permitted fixed pair placed 58/58 costed operations on ANE. |
| G0d: pair numerics | **KILL** | All three pairs exceeded `1e-2`; pair 6-7 changed the top token. |
| G0c: aggregate pair value | Cancelled | Correctness failed before timing, by protocol. |

The gate labels list G0c before G0d, but the frozen ground-truth contract says
"Correctness before speed." G0d therefore runs as the preflight to any G0c
latency sample; the labels and thresholds themselves were not changed.

### G0a: same-package monolith scheduler

Both rows use the fixed-512 monolithic layer-stack package with SHA-256
`e57a6aae9ffd2bf375cc237f64ab8cee6ddea4ea15d27d1011f6fbec8a855d54`.
Embedding and head remained `.cpuAndGPU`.

| Policy | Median | IQR | Preferred placement | Top token |
| --- | ---: | ---: | --- | ---: |
| `M_GPU` | 35.539 ms | 0.989 ms | 784/784 GPU | 941 |
| `M_ALL` | 53.281 ms | 1.952 ms | 308/784 ANE, 476/784 GPU | 1470 |

The registered difference was `median(M_GPU) - median(M_ALL) = -17.742 ms`.
Its deterministic 100,000-resample paired 95% interval was
`[-18.694, -17.065] ms`, wholly favoring `M_GPU`. Final-logit maximum absolute
error was `10.0859375`. The `.all` plan did place 78/102 convolution operations
on ANE while all 12 matmuls and all 6 softmaxes remained on GPU, but placement
alone did not produce a valid or faster result.

### G0b and G0d: fixed-512 convolution pairs

`source max_abs` compares the original enumerated package at bucket 512 with
its materialized fixed twin under `.cpuAndGPU`. `GPU↔ANE max_abs` covers the
hidden output and both convolution-state outputs. `suffix max_abs` compares
final logits after the common untimed GPU suffix and head.

| Pair | Fixed package SHA-256 | ANE admission | Source max_abs | GPU↔ANE max_abs | Suffix max_abs | GPU→ANE top token | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| C0-1 | `dfbc10cad5d42c26b97b69d1ce74f01a28a07f170323529871aa4a4fb8576218` | 58/58 | 0.0 | 0.288086 | 7.826172 | 941→941 | Fail |
| C3-4 | `db264232357277efdc5d0efca961c50ba1971e53018da3efade7b29583e7133a` | 58/58 | 0.0 | 0.154297 | 9.095703 | 941→941 | Fail |
| C6-7 | `fd270492253c5f653ead28ff0368ae912278ae21fb201a9910d97c337be29b5b` | 58/58 | 0.0 | 0.073975 | 3.377930 | 941→509 | Fail |

The zero source error for every pair isolates the failure from shape
materialization. The same fixed package and the same real activation were used
for GPU- and ANE-permitted execution. Because every pair failed correctness,
no pair latency sample was collected or reported.

## Reproduce Phase 0

Use the frozen Python environment described in the Stage 0 report. The local
run used `/Users/mm/Documents/GitHub/kokoro-coreml/.venv/bin/python` because it
already contained the exact Python 3.11, coremltools 8.3.0, PyTorch 2.6.0, and
NumPy 1.26.4 stack.

```bash
python scripts/lfm2_surgical/materialize_fixed_pairs.py \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --out-dir outputs/lfm2_surgical/selective/fixed_pairs_512

python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage1/embedding.mlpackage \
  --compute-units CPU_AND_GPU \
  --out outputs/lfm2_surgical/selective/phase0_dispatch/embedding_gpu.json

python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage1/lm_head.mlpackage \
  --compute-units CPU_AND_GPU \
  --out outputs/lfm2_surgical/selective/phase0_dispatch/lm_head_gpu.json

python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage1/monolithic_layer_stack_512.mlpackage \
  --compute-units CPU_AND_GPU \
  --out outputs/lfm2_surgical/selective/phase0_dispatch/monolithic_gpu.json

python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage1/monolithic_layer_stack_512.mlpackage \
  --compute-units ALL \
  --out outputs/lfm2_surgical/selective/phase0_dispatch/monolithic_all.json

for pair in segment_00_conv_0_1 segment_02_conv_3_4 segment_04_conv_6_7; do
  python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/selective/fixed_pairs_512/$pair.mlpackage" \
    --compute-units CPU_AND_GPU \
    --out "outputs/lfm2_surgical/selective/phase0_dispatch/${pair}_fixed512_gpu.json"
  python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/selective/fixed_pairs_512/$pair.mlpackage" \
    --compute-units CPU_AND_NE \
    --out "outputs/lfm2_surgical/selective/phase0_dispatch/${pair}_fixed512_ane.json"
done

swift build -c release
.build/release/lfm2-selective-benchmark phase0 \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --input-json outputs/lfm2_surgical/stage1/g1a_input_512.json \
  --dispatch-dir outputs/lfm2_surgical/selective/phase0_dispatch \
  --fixed-pair-dir outputs/lfm2_surgical/selective/fixed_pairs_512 \
  --warmups 3 --runs 20 \
  --out outputs/lfm2_surgical/selective/phase0_mac.json
```

Durable generated evidence is ignored by Git and remains at:

- `outputs/lfm2_surgical/selective/fixed_pairs_512/manifest.json`
- `outputs/lfm2_surgical/selective/phase0_dispatch/embedding_gpu.json`
- `outputs/lfm2_surgical/selective/phase0_dispatch/lm_head_gpu.json`
- `outputs/lfm2_surgical/selective/phase0_dispatch/monolithic_{gpu,all}.json`
- `outputs/lfm2_surgical/selective/phase0_dispatch/*_fixed512_{gpu,ane}.json`
- `outputs/lfm2_surgical/selective/phase0_mac.json`
- `outputs/lfm2_surgical/selective/stage1_replay.json`

The frozen Stage 1 replay remained bit-exact across its public fp16 boundaries
and again returned `KILL`: monolith `35.432 ms`, 13 segments `57.449 ms`,
`38.32%` decomposition tax, and final-logit `max_abs = 0.0`.

## Limitations

- This is one real prompt, one 512-token bucket, one M2 Ultra host, and one OS
  build. It is a falsification of this registered path, not a universal claim
  that every LFM2 convolution is numerically unsafe on every ANE generation.
- Compute-plan preference proves admission, not execution speed or accuracy.
- No pair speed, six-piece end-to-end speed, device transfer, energy, decode,
  thermal-soak, iPad, or iPhone claim is made.
- Generated checkpoint-derived packages are not published here. They remain
  governed by the LFM Open License v1.0.

The simplest conclusion is the right one: do not pay engineering or device-lab
cost for a partition whose claimed accelerator islands already fail numerical
equivalence on the Mac gate.
