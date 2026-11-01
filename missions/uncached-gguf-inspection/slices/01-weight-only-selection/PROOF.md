# PROOF — SYN.99.0.1.1 Weight-only uncached selection and bounded hover

Recorded by dev-build@starter on 2026-10-26. Candidate f22f2239d489acaad42754c8e0955d5e188ea4a9. Delivered with an explicit full-inference baseline verification residue. Independent reviewer found no blocking product defect. Parent deployed, exercised the real browser, and confirmed publication.

## What this proves

Uncached links, labels and webs work below inference admission using this GGUF's read-only weights without a generation context. Rapid selection changes keep the latest result, including delayed A→B→A; actual pointer hover causes no heavy scans. The reviewed revision is running on live8791 and published to origin/main. The inference guard was retained; a fresh before/after inference-logit comparison was not run.

## Contract mapping

Paths below are relative to /home/park/Projects/synapse unless absolute.

| Item | Observed outcome and evidence |
| --- | --- |
| 1 — cold below-admission inspection | PASS. Builder empty-cache eight selections include every reported L31 failure, plus early/middle layers; inference forbidden, 3367.875 MiB available, peak RSS 469.535 MiB. .synapse/verification/inspection-y4itj1py/result.json. Reviewer repeated three layers in another empty cache at 3165.602 MiB available, peak 468.465 MiB: evidence/review-f22f223-native.json. |
| 2 — browser/races/hover after restart | PASS. Parent deployment then six initially absent browser coordinates: evidence/uncached-parent-acceptance.json and .log. Reviewer-authored delayed ABA and pointer-hover script executed externally by parent: evidence/review-f22f223-browser.json, -browser-external.log and -browser-addendum.md. Latest marker2 survived; focus/lens/web agreed; six hover requests caused no scan routes or JS errors. |
| 3 — numerical/lens/baseline/hash | PARTIAL. Independent NumPy vector errors below 1.7e-9, scalar/batched labels and raw repeatability agree; model SHA256 unchanged. evidence/review-f22f223-native.json. Fresh inference-path comparison and before/after baseline logits were NOT rerun; admission was not bypassed. Earlier baseline receipts are historical only. |
| 4 — resources/failure controls | PASS for measured operation. No-context inference rejects; invalid coordinates422; hover leaves443 cache rows unchanged: evidence/review-f22f223-http.json. Initial cold-browser cgroup peak1296687104 bytes (1.21 GiB); later lifetime peak1733156864 bytes (1.61 GiB), same invocation, swap0, active under5 GiB/no swap. Initial: evidence/uncached-parent-acceptance.json and uncached-parent-deployment.json. Later: /home/park/.local/state/synapse-uncached-repair-20261026/operator-deployed-service-check.json. Separate chat unit inactive. |
| 5 — regression/independent verdict | PASS. Builder and reviewer each32 tests; exact nine-file diff reviewed, no correction requested. evidence/gguf-inspection-regressions.txt, review-f22f223-pytest.txt, review-f22f223-verdict.md and -browser-addendum.md. Raw initial native failure, interrupted run and sandbox browser failures retained. |
| 6 — deploy/publish/delivery | PASS for deploy/publish and post-deploy effect. Exact tree/library binding: evidence/uncached-parent-deployment.json. git push succeeded and git ls-remote confirmed f22f223: evidence/uncached-repair-publication.json. Review-return qitem-20261026114915-2a358248 transfers evidence/lithos-uncached-repair-result.md to operator for completion receipt; parent independently confirmed remote hash. |

## Artifacts

Native proof drops: proof/review-f22f223.md, the reviewer's browser addendum, and proof/delivery-f22f223.md. The delivery drop covers items1,2,4,5,6 only; it does not attest that all of item3 passed. Attached proof/uncached-parent-browser.png shows the final production selection, labels, neighbor list and map. Detailed receipts are retained at their existing evidence paths.

## Residue and continuation

The only remaining numerical verification limit is fresh full-inference equivalence/baseline logits. The optional scripts/verify_gguf_inspection.py --reference .synapse/verification/inspection-y4itj1py run remains behind production admission and the shared lease, with model services stopped and bounded memory. Run only when coordinated memory permits. Parent/operator explicitly accepted recording this unrun limit; no additional product edits or test runs were requested for this delivery.

Sandbox browser failures are preserved, including the proven WebGL2 failure. Successful browser execution was performed externally by parent, with a reviewer-authored supplemental test and independent inspection of its output. Resource values use native RUSAGE or host systemd/smaps/cgroup; namespace-local test PIDs are not cross-process memory evidence. Test fixtures/drivers stopped; owner Waterfox and unrelated apps preserved. Models, caches, credentials, launch files and machine records are outside the published commit.
