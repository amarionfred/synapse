---
id: SYN.99.0.1.1
slice: 01-weight-only-selection
mission: uncached-gguf-inspection
status: delivered-with-verification-residue
stage: established
verified: 2026-10-26 against f22f223 independent review, parent deployment/browser receipts and remote hash; PROOF.md records the full-inference baseline limit
created: 2026-10-26
intent: "Separate GGUF weight inspection from inference loading and verify cold-cache navigation, request races, numerics and resource bounds."
depends_on: []
---

# Slice 01 — Weight-only uncached selection and bounded hover

## Intent

Separate GGUF weight inspection from inference loading, then verify fresh selections, request races, native numerics and measured resource bounds in the real user path.

## Mini-requirements

1. Cold connections/lens/web for supported early, middle and last-layer neurons work with inference deliberately unavailable and RAM below its admission threshold. Use the checkpoint's metadata/tokenizer/tensors, not borrowed caches. Preserve the inference guard and shared inference slot.
2. Rapid selection changes clear old selection-owned data and cannot be overwritten by late responses. Root/final empty sides remain valid. Hover is bounded and cache-only; handle errors/cancellation and avoid competing scans.
3. Preserve native numerical behavior, baseline inference logits and GGUF SHA256 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4. Inspect pinned llama.cpp loader support before choosing implementation.
4. Keep the service at its 5 GiB/no-swap bound. Serialize memory-heavy validation. No duplicate full models, saved-weight mutation, broad sweeps, new model/dependency downloads, closing Waterfox or unrelated application stops. /tmp is RAM-backed; artifacts belong in project/.synapse or existing evidence paths.
5. Update existing DECISIONS.md, README/capabilities and evidence/synapse-verification.json honestly. Return exact commit/files, raw checks/resource receipts, independent verdict and limits in evidence/lithos-uncached-repair-result.md and the native queue. Push only the authorized repair; exclude models, credentials, caches, chats and machine-specific launch records.

## Proof contract

- [x] Reproduction with separate/empty inspection cache, inference unloaded and insufficient inference-admission RAM succeeds for early/middle/final neurons, including L31 user coordinates 7066, 7621, 4089, 2565, 6399 and 7166.
- [x] Actual browser rapid switching, hover movement and new uncached coordinates after restart show the selected neuron's own results; no stale overwrite or queued hover weight scans.
- [ ] Numerical score/lens agreement and unchanged baseline logits/model hash are established against the validated path, with resource-heavy checks serialized.
- [x] Inspection-only resources and failure behavior are measured; service resource controls and inference admission remain intact.
- [x] Relevant regressions and one independent cold-cache/race-focused review pass on the final candidate; raw failures, corrections and limitations are retained.
- [x] Local deployed service and authorized remote commit are identified; actual fresh-cache user path is checked after deployment and result delivered to operator for external acceptance.

Item3 remains partial: independent native numerical/lens agreement and unchanged model hash passed; fresh full-inference comparison/baseline logits were not rerun because the admission guard was retained. Historical baseline receipts are not substituted. See PROOF.md for the exact mapping and retained limitation.

## Repro

See evidence/gguf-uncached-failure.json and ../../AUTHORITY.md. The prewarmed L14 N8949 path masks the defect; failing L31 coordinates return 422 requiring 4352 MiB before inference load with about 4135 MiB available.

## Expected and actual

Expected: inspecting weights does not need an inference context. Actual: ensure_loaded on cache misses reaches syn_open, which loads model plus a 2048-token generation context. Hover and stale-selection requests also need review.

## Impact

The user's ordinary neuron navigation fails. Prior warmed-neighborhood checks and 29 regressions did not establish this delivery. A helpful 422 or lowering the RAM guard is not the requested fix.
