# A17 Pro Parity Report: The ANE Numerical Failure Reproduces

**Date:** 2026-07-21

**Status:** Terminal `KILL` at preregistered Gate P2

**Checkpoint:** `LiquidAI/LFM2.5-350M` at
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`

**Device:** iPhone 15 Pro Max (`iPhone16,2`, A17 Pro), iOS 27.0 beta
(`24A5380h`)

**Runtime:** Signed Release Swift/Core ML app, direct fp16 `MLMultiArray`, no
performance samples

## Decision

Stop work on this public-boundary selective-prefill direction. The fixed-512
C6-7 package was admitted completely to the A17 Pro Neural Engine, but its
ANE-permitted output violated the frozen `1e-2` numerical tolerance in every
first-use and warmed comparison. The common GPU suffix amplified the error and
the final token changed from `941` to `509`.

This is the same decisive failure pattern previously observed on M2 Ultra. The
registered C0-1 and C3-4 phone probes were therefore cancelled before their
compute plans or apps were built. No latency, duration, throughput, energy, or
thermal-performance sample was collected.

## Frozen question

Plan 012 asked one narrow question: does the fixed-512 ANE numerical failure
seen on M2 Ultra reproduce on an A17 Pro iPhone?

The answer is yes for the highest-information C6-7 probe. The experiment kept
the checkpoint, real prompt, bucket 512, fp16 precision, zero prefill state,
fixed package, GPU control, ANE-permitted candidate, and `1e-2` maximum-absolute
tolerance unchanged. It recorded one first-use comparison followed by five
warmed comparisons in the preregistered alternating call order.

## Signed device and artifact identity

The host revalidated the physical phone immediately around device operations:

- CoreDevice identifier: `8A12AEE8-0136-50BE-8EB3-91650E467F15`
- hardware UDID: `00008130-000600660EA2001C`
- product type: `iPhone16,2`
- iOS version/build: 27.0 beta (`24A5380h`)
- app identifier: `6ETYBAJKY8.com.mattmireles.LFM2A17Parity`
- explicit profile UUID: `173c72a0-1f63-4748-95bd-bd195ca1580f`
- fixed C6-7 package SHA-256:
  `fd270492253c5f653ead28ff0368ae912278ae21fb201a9910d97c337be29b5b`

The result records the exact executed app revision as public commit `a30820b`
plus worktree fingerprint
`5ba1a9a4cb6eaf683a2d340f42b0447ca83c82cbea9b9e6f724fba939867060d`.
That worktree delta is the explicit signing-identity repair required by the
Core ML model-runner helper; it changed no model or numerical variable.

## Same-phone dispatch gate

Both plans were captured on the result-producing phone and OS build before the
parity app launched. Each dispatch file is bound to the exact fixed-package
hash.

| Policy | Preferred-device histogram | Convolution placement | Result | Evidence file SHA-256 |
| --- | --- | --- | --- | --- |
| `CPU_AND_GPU` | 54 GPU, 2 CPU, 0 ANE | 12/12 GPU | Control passed | `6855d7ec08195b63a05a5b566f666b187ef254d596f810f0961d9e1371f9a05d` |
| `CPU_AND_NE` | 56 ANE | 12/12 ANE | Admission passed | `1a3bb1f7143501fc8971bd0abb9f4e9142414e5bbc4e32032b8f9d32bd571470` |

The two CPU-preferred control operations were one `reduce_mean` and one `pow`.
The ANE-permitted candidate placed all 56 costed operations on ANE. The failure
therefore cannot be explained by both policies accidentally selecting the
same backend.

## Untimed correctness result

The complete all-GPU source oracle passed with token `941`. The original
enumerated C6-7 source and its fixed twin matched exactly under
`CPU_AND_GPU` in all six rows (`source max_abs = 0`). Only the compute policy
changed between the fixed GPU and fixed ANE-permitted candidates.

| Row | Kind | Call order | Source↔twin max_abs | GPU↔ANE max_abs | Suffix-logit max_abs | GPU→ANE token | Result |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | First use | GPU→ANE | 0 | 0.070068 | 3.610229 | 941→509 | Fail |
| 1 | Warmed | GPU→ANE | 0 | 0.070068 | 3.610229 | 941→509 | Fail |
| 2 | Warmed | ANE→GPU | 0 | 0.070068 | 3.610229 | 941→509 | Fail |
| 3 | Warmed | GPU→ANE | 0 | 0.070068 | 3.610229 | 941→509 | Fail |
| 4 | Warmed | ANE→GPU | 0 | 0.070068 | 3.610229 | 941→509 | Fail |
| 5 | Warmed | GPU→ANE | 0 | 0.070068 | 3.610229 | 941→509 | Fail |

The per-output GPU-versus-ANE errors were identical in every row:

| Named output | max_abs | Frozen tolerance | Result |
| --- | ---: | ---: | --- |
| `hidden_out` | 0.012085 | 0.010000 | Fail |
| `conv_state_6_out` | 0.000977 | 0.010000 | Pass |
| `conv_state_7_out` | 0.070068 | 0.010000 | Fail |

First-use and warmed results were identical, and reversing candidate order did
not change them. This is a deterministic policy-dependent numerical failure,
not a warmup or mutable-output artifact.

## Durable evidence

Generated evidence remains ignored because it includes derived checkpoint
packages and device-local artifacts. The result-producing checkout retains:

- `outputs/lfm2_surgical/a17_parity/dispatch/segment_04_conv_6_7.cpu_and_gpu.json`
- `outputs/lfm2_surgical/a17_parity/dispatch/segment_04_conv_6_7.cpu_and_ne.json`
- `outputs/lfm2_surgical/a17_parity/pairs/segment_04_conv_6_7.result.json`

The result envelope file SHA-256 is
`6878e45d085ea0683c5f8ee864e13eb5a25f228f7a3245155c28b74ea44f5568`.
Its canonical payload SHA-256 is
`53fcae5d2502d6e9aae93fe1c85d0523ae378ae126a6ea877611308e574f6c20`.
The schema records `performance_samples_recorded = 0`.

The executed commands were:

```bash
python scripts/run_a17_parity.py dispatch \
  --pair segment_04_conv_6_7 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid 173c72a0-1f63-4748-95bd-bd195ca1580f \
  --out-dir outputs/lfm2_surgical/a17_parity/dispatch

python scripts/run_a17_parity.py run \
  --pair segment_04_conv_6_7 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid 173c72a0-1f63-4748-95bd-bd195ca1580f \
  --out outputs/lfm2_surgical/a17_parity/pairs/segment_04_conv_6_7.result.json
```

## Interpretation and limits

The A17 result narrows the explanation: the plan-011 failure is not confined
to the tested M2 Ultra ANE/compiler path. The same fixed package, prompt, real
activation, and token-changing failure reproduced on the tested A17 Pro and
iOS build after complete ANE admission.

This does not prove that every LFM2 convolution package fails on every Apple
Neural Engine, precision, shape, model, or OS. It does falsify the registered
claim that this exact public-boundary fixed-512 island is a valid GPU/ANE
substitute on the tested phone. Testing more devices merely to search for a
pass would not repair that claim.

The simplest conclusion is the right one: stop this partitioning approach.
A future experiment requires a materially new mechanism and a new
preregistration—not a looser tolerance, another device, or post-hoc timing.
