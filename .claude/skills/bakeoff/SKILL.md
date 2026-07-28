---
name: bakeoff
description: >-
  Run a controlled LFM2 surgical-inference comparison only when a new
  experiment is explicitly approved. Default to the recorded terminal KILL:
  37.43% segmentation prediction overhead at 512-token prefill. Do not restart
  the cancelled matrix implicitly.
---

# LFM2 bakeoff

Before any run, require a new hypothesis, frozen checkpoint, prompt/bucket
matrix, baseline, variants, package hashes, metrics, threshold, device, and stop
rule that explicitly supersede or investigate the terminal result. Otherwise
report the existing KILL and stop. Negative evidence is the completed outcome,
not unfinished work.
