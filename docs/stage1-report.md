# Stage 1 Report: Decomposition Tax Kills the 13-Segment Pipeline

**Date:** 2026-07-20  
**Checkpoint:** `LiquidAI/LFM2.5-350M` at
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`  
**Verdict:** **KILL at G1a**  
**Continuation:** Phases 4-5 and the iPhone matrix are cancelled

## Result

The segmented LFM2.5 layer stack is numerically identical to the monolithic
control at its final hidden output, and every costed operation in both controls
prefers the GPU on the measurement Mac. It is still much slower.

At the frozen 512-token gate, the monolithic end-to-end prefill path takes a
median `36.032 ms`. The 13-segment path takes `57.591 ms`. The `21.558 ms`
boundary tax consumes `37.43%` of segmented prefill, above the
pre-registered `30%` kill threshold.

| 512-token GPU-permitted control | Median | IQR | Costed GPU ops |
| --- | ---: | ---: | ---: |
| Monolithic full prefill | 36.032 ms | 0.352 ms | 797/797 |
| 13-segment full prefill | 57.591 ms | 4.367 ms | 807/807 |
| Boundary tax | **21.558 ms** | — | — |

Two equivalent views of the tax are useful:

- `(segmented - monolithic) / segmented = 37.43%` — the registered G1a
  definition, because the plan caps boundary work at 30% of total segmented
  prefill.
- `(segmented - monolithic) / monolithic = 59.83%` — segmentation makes the
  full prefill path 1.60× as slow as the no-boundary control.

The result is not a placement or language-bridge confound. `MLComputePlan`
reports 797/797 GPU operations for the monolithic full path and 807/807 GPU
operations for the segmented full path. A release Swift executable hands
fp16 `MLMultiArray` outputs directly between models—no Python or NumPy boundary
exists in the authoritative timing. The paired final logits match exactly
(`max_abs = 0.0`) in all 20 measured pairs.

An independent second Swift N=20 run also failed G1a: `37.292 ms` monolithic,
`59.771 ms` segmented, and `37.61%` of segmented prefill consumed by the tax.

## What Stage 1 built

The real 16-layer checkpoint order

```text
C C A C C A C C A C A C A C A C
```

becomes 13 maximal same-class runs:

| Segment | Layers | Class | Explicit outputs |
| --- | --- | --- | --- |
| 00 | 0-1 | conv | hidden + states 0, 1 |
| 01 | 2 | GQA | hidden + K/V 2 |
| 02 | 3-4 | conv | hidden + states 3, 4 |
| 03 | 5 | GQA | hidden + K/V 5 |
| 04 | 6-7 | conv | hidden + states 6, 7 |
| 05 | 8 | GQA | hidden + K/V 8 |
| 06 | 9 | conv | hidden + state 9 |
| 07 | 10 | GQA | hidden + K/V 10 |
| 08 | 11 | conv | hidden + state 11 |
| 09 | 12 | GQA | hidden + K/V 12 |
| 10 | 13 | conv | hidden + state 13 |
| 11 | 14 | GQA | hidden + K/V 14 |
| 12 | 15 | conv | hidden + state 15 |

Every segment uses the frozen enumerated prompt shapes `{128, 256, 512, 1024,
2048}`. Each conv layer accepts and emits its own explicit `[1, 1024, 2]`
rolling state. Every attention layer emits its `[1, 8, S, 64]` K/V seed. The
embedding and one-position tied LM head are separate packages. The fixed-512
monolithic control uses the same equations, fp16 weights, mixed-precision
islands, state inputs, and state outputs without prediction boundaries.

One full segmented prefill on a real prompt chose token `509`, matching the
independent fp32 16-layer reference. That probe is supportive, not the frozen
G1b equivalence result. The shared Swift orchestrator was completed to make
G1a authoritative. The planned 32-prompt × 64-token G1b matrix was not run
because G1a independently requires the experiment to stop.

## Protocol

- Host: Mac Studio M2 Ultra (`M2-Studio.local`)
- macOS: 26.5.2
- coremltools: 8.3.0
- Compute-unit permission: `CPU_AND_GPU`
- Bucket: 512 tokens
- Prompt: the first frozen real prompt
- Warmup: three predictions per candidate
- Measurement: 20 predictions per candidate
- Order: counterbalanced `AB/BA` on every successive pair
- Excluded: package loading and first-use compilation
- Runtime: release Swift executable using direct fp16 `MLMultiArray` handoffs
- Included in both controls: embedding, layer stack, last-position copy, and
  tied LM head
- Included only in the segmented control: all 13 layer-stack `prediction()`
  calls, 12 added inter-segment hidden boundaries, explicit conv state outputs,
  and GQA K/V outputs

The earlier Python diagnostic exposed fp16 model outputs as fp32 NumPy arrays
and measured a larger 54.75% tax. That result is retained in
`decomposition_tax.json` but does not govern the gate. The Swift runtime
records `hiddenBoundaryDataType = float16` and directly reuses each output
`MLMultiArray` as the next model's input.

The runtime prints `tensor_buffer has known strides while the model has
FlexibleShapeInfo` for flexible conv segments with fixed-size state inputs.
Predictions, exact output comparison, and compute-plan inspection all complete
successfully. This warning is retained as evidence; it is not used to explain
away or subtract any latency.

## Dispatch evidence

| Artifact | Preferred device | Costed operations |
| --- | --- | ---: |
| Embedding (common) | GPU | 4/4 |
| LM head (common) | GPU | 9/9 |
| Monolithic 512 layer stack | GPU | 784/784 |
| Conv segment 00 | GPU | 58/58 |
| GQA segment 01 | GPU | 84/84 |
| Conv segment 02 | GPU | 58/58 |
| GQA segment 03 | GPU | 84/84 |
| Conv segment 04 | GPU | 58/58 |
| GQA segment 05 | GPU | 84/84 |
| Conv segment 06 | GPU | 29/29 |
| GQA segment 07 | GPU | 84/84 |
| Conv segment 08 | GPU | 29/29 |
| GQA segment 09 | GPU | 84/84 |
| Conv segment 10 | GPU | 29/29 |
| GQA segment 11 | GPU | 84/84 |
| Conv segment 12 | GPU | 29/29 |
| **Segment total** | **GPU** | **794/794** |

Including the common embedding and LM head gives 797/797 GPU operations for
the monolithic full path and 807/807 for the segmented full path.

Each durable dispatch JSON includes the exact package-tree SHA-256 and the
host compute-unit policy. The benchmark JSON binds the embedding, LM head,
monolith, and all 13 segments to their package-tree hashes.

## Reproduce

```bash
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope segments
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope embedding
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope head
.venv/bin/python scripts/lfm2_surgical/export_segments.py \
  --all-buckets --scope monolith

.venv/bin/python scripts/lfm2_surgical/benchmark_decomposition_tax.py \
  --warmups 3 --runs 20

.venv/bin/python scripts/lfm2_surgical/prepare_stage1_input.py
swift build -c release
.build/release/lfm2-surgical-benchmark \
  --stage1-dir outputs/lfm2_surgical/stage1 \
  --input-json outputs/lfm2_surgical/stage1/g1a_input_512.json \
  --warmups 3 --runs 20 \
  --out outputs/lfm2_surgical/stage1/decomposition_tax_swift.json
```

Both benchmark executables exit nonzero by design when G1a fails, after
writing their JSON evidence. `decomposition_tax_swift.json` is authoritative;
the Python result is a diagnostic proving why runtime boundaries matter.

Generate the same-device dispatch evidence:

```bash
for name in embedding lm_head; do
  .venv/bin/python scripts/dump_device_compute_plan.py \
    --package "outputs/lfm2_surgical/stage1/${name}.mlpackage" \
    --compute-units CPU_AND_GPU \
    --out "outputs/lfm2_surgical/stage1/dispatch_mac/${name}.json"
done

.venv/bin/python scripts/dump_device_compute_plan.py \
  --package outputs/lfm2_surgical/stage1/monolithic_layer_stack_512.mlpackage \
  --compute-units CPU_AND_GPU \
  --out outputs/lfm2_surgical/stage1/dispatch_mac/monolithic_512.json

for package in outputs/lfm2_surgical/stage1/segments/*.mlpackage; do
  name="$(basename "$package" .mlpackage)"
  .venv/bin/python scripts/dump_device_compute_plan.py \
    --package "$package" \
    --compute-units CPU_AND_GPU \
    --out "outputs/lfm2_surgical/stage1/dispatch_mac/${name}.json"
done
```

Generated packages and JSON evidence remain uncommitted because they include
about 1.3 GB of derived model artifacts governed by the LFM Open License.

## Decision

Stop. Do not run the M1, iPad, or iPhone matrices. The experiment froze G1a
specifically to prevent attractive per-block dispatch from justifying a
pipeline whose boundaries already consume most of its prefill time.

This result narrows the Surgical Inference claim: operator-class placement is
not enough when the architecture interleaves classes so finely that it creates
13 segment predictions and 12 added boundaries. LFM2.5-350M needs a
lower-boundary packaging or
runtime mechanism before heterogeneous placement can be competitive. Testing
multifunction packaging or a fused runtime is follow-up work, not a rescue of
this pre-registered experiment.
