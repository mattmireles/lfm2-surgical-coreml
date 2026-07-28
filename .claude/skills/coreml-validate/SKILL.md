---
name: coreml-validate
description: Validate LFM2.5 surgical Core ML block outputs against the frozen reference across the 32 registered prompts and five shape buckets. Use for artifact maintenance or an explicitly approved follow-up; never weaken frozen parity gates.
---

# Validate LFM2 surgical Core ML

Freeze checkpoint revision, independent fp32 oracle, prompts, bucket shapes,
package hashes, dtypes, and `1e-2` max-absolute error gate. Compare every
registered output cell and preserve the exact Stage 0 evidence. Validation does
not authorize Stage 1 continuation; the recorded segmentation-overhead KILL
still governs.
