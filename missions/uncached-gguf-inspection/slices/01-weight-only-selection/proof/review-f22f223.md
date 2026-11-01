---
slice: SYN.99.0.1.1
candidate_sha: f22f2239d489acaad42754c8e0955d5e188ea4a9
artifact_type: qa
verdict: PASS
money_evidence: Independent private-cache GGUF inspection passed below inference
  admission at 468.465 MiB peak RSS; NumPy vectors, model hash, hover controls
  and 32 regressions passed. Parent production cold and rapid browser receipt
  inspected; full-inference and forced ABA browser limits retained.
evidences:
  - "1"
  - "5"
self_check: I ran the reviewer checks, inspected exact outputs and the parent
  browser screenshot/receipt, and recorded unexecuted coverage without asserting
  it passed.
---

# Independent review of f22f223

Reviewer: dev-review@starter, 2026-10-26. Candidate: f22f2239d489acaad42754c8e0955d5e188ea4a9, parent c4c270b1533fbcd03e4fe0f968da03dae6e14ef5. Complete nine-file product diff reviewed; final git diff against the candidate was empty for those files. Reviewer authored only ignored verification artifacts, no product changes.

Verdict: **PASS for the candidate repair, with the verification limits below. No blocking product defect found.** This verdict does not close deployment documentation, publication, or the whole slice.

## Direct reviewer evidence

- The documented test command passed: 32 tests, 2 existing FastAPI deprecation warnings, 17.38 seconds. Raw output: evidence/review-f22f223-pytest.txt.
- A new private inspection cache was initially empty. L0 N113, L15 N417 and L31 N7066 returned their own connections and positive/negative lens labels with ensure_loaded replaced by a failing function. The first/final empty sides were correct. No inference model, tokenizer slot, native context or inference lease was created. Before inspection, cached previews did not create an inspector. Receipt: evidence/review-f22f223-native.json; command: .venv/bin/python evidence/review_f22f223_native.py; log: evidence/review-f22f223-native-v2.log.
- Available RAM was 3165.602 MiB, below the 4352 MiB inference admission requirement. Process peak RSS was 468.465 MiB. The three selections, independent matrix checks, scalar lens and model hash completed in 59.42 seconds. These numbers measure this bounded check, not a general latency guarantee.
- Independent NumPy float64 matrix products for all 9216 upstream/downstream neurons in adjacent layers agreed with the native vectors: maximum absolute errors 8.77e-10 and 1.69e-9. Scalar/batched top-12 labels agreed. Raw vectors and scalar lens exactly repeated the builder's saved reference. Native evaluation on the inspection handle rejected with “Weight-only inspection has no inference context”.
- GGUF SHA256 remained 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4. Tested native library SHA256: f23695b8866ef4928c7f4921a822da363d477e35c59ec05d5dbc5a2a0d449f86.
- Live production HTTP controls passed after the parent's deployment: three uncached hover requests returned empty pending previews; cache row count remained 443 before/after. Invalid layer -1, layer32 and neuron9216 returned 422. Inference stayed unloaded. Receipt: evidence/review-f22f223-http.json; command: .venv/bin/python evidence/review_f22f223_http.py --production. This establishes endpoint behavior; it does not claim actual pointer-hover coverage.
- Source review confirms read-only mmap, vocab_only model loading, bounded dequantization/page release and an inspector without inference context. The production inference admission and shared lease remain; the inspector closes before inference creation. Selection request ownership uses controller identity, including A-to-B-to-A. The server's inspection gate retains ownership until a running worker finishes and skips queued disconnected callers. Hover calls cached methods only. The cancellation/worker test passed in the regression suite.

## Evidence inspected from other evaluators

- Builder's .synapse/verification/inspection-y4itj1py/result.json contains eight successful cold selections, including all six user-reported L31 failures. Their connection rows were uncached, labels present, and inference false throughout. Its recorded available RAM was 3367.875 MiB and peak RSS 469.535 MiB. I inspected these values; I independently reran three of these coordinates in a different empty cache.
- Parent's production evidence/uncached-parent-acceptance.json and .log record absent cache entries before six distinct selections, three sequential early/middle/final selections, then rapid cross-layer switching. The selected lens and web coordinates match focus, first/final side counts are correct, and final links/labels match the final requested selection per the parent's acceptance assertions. No unhandled JavaScript errors were recorded. Inference remained false; inspection changed from false to true.
- I visually inspected evidence/uncached-parent-browser.png: the selected-neuron name, token labels, neighbor list and map render. This is the parent's browser run, not a reviewer-controlled run.
- Parent's resource receipt records service cgroup peak 1296687104 bytes (1236.617 MiB, including file cache), final RSS 526784 KiB, zero service swap, and initial available RAM 3516496 KiB. This differs from standalone process RSS by design.

## Verification limits and retained failures

1. Full-inference reference comparison and before/after baseline logits were not rerun for this candidate; the normal admission gate was preserved and never bypassed. Prior inference-path evidence and source continuity support implementation confidence, but are not a new baseline measurement. Keep this limitation in the final result; execute --reference only when admission and coordinated memory permit it.
2. My owned Firefox session could not initialize WebGL2 in the sandbox. The actual page's geckodriver log reports Three.js “Error creating WebGL context”, and a separate fresh-canvas check returned false. The diagnostic dynamic import additionally reported an unmapped bare specifier, so that import alone is not the graphics diagnosis. Receipt/log: evidence/review-f22f223-browser.json, evidence/review-f22f223-geckodriver.log. The production parent browser rendered successfully, so this is a reviewer-environment limitation, not a reproduced deployed defect.
3. Consequently the reviewer's forced delayed A-to-B-to-A and actual pointer-hover checks did not execute. Production rapid A-to-B-to-C switching passed in the parent's real browser; cache-only hover passed by endpoint, test and source review. If the final contract requires the stronger two browser scenarios, run the prepared evidence/review_f22f223_browser.py outside this sandbox using an owned geckodriver4448, while respecting the same resource serialization. The script records actual hover request count, which must be nonzero to claim pointer-hover coverage.
4. Live systemd controls could not be independently queried from this sandbox (user bus inaccessible, including an explicit /run/user/1000 bus). The unchanged launcher specifies 5 GiB/no swap; parent/operator must retain their live-control receipt and separate-chat stopped status.
5. Earlier interrupted native work and the intentionally stopped fixture's connection refusal are retained in evidence/review-f22f223-interruptions.md and the first native log. Browser session was DELETEd and reviewer-owned geckodriver4448 stopped after diagnosis. No owner application was stopped, no full inference started, no model weights changed, no new dependencies downloaded, and no reserved acceptance script was read or run. Parent results were inspected only after it completed and supplied them for review.

## Return to builder

Integrate this verdict and the parent production receipt into the canonical result and final documentation. Preserve the explicit baseline/browser limits. Complete authorized publication and identify the deployed/published revision through the existing queue. No product correction is requested by this review.
