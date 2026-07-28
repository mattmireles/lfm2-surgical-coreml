---
name: coreml-profile
description: Profile LFM2.5 surgical Core ML block packages and device compute plans. Use only for maintaining or explicitly extending the completed experiment. Bind results to package hashes, prompt buckets, hardware, OS, and compute-unit requests; preserve the recorded terminal KILL.
---

# Profile LFM2 surgical Core ML

Use `scripts/dump_device_compute_plan.py` and the frozen five-bucket package
matrix. Record package-tree hashes and physical-device evidence. Distinguish
requested from admitted compute units. The existing Stage 1 result—37.43%
segmentation prediction overhead at 512-token prefill—is terminal; do not
launch the cancelled device matrix without a new explicitly approved
preregistered experiment.
