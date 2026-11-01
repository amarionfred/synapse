---
slice: SYN.99.0.1.1
candidate_sha: f22f2239d489acaad42754c8e0955d5e188ea4a9
artifact_type: qa
verdict: PASS
money_evidence: External execution of reviewer-authored browser probe preserved
  marker2 after delayed ABA and produced six actual hover requests with zero
  scan routes or JavaScript errors; reviewer validated receipt.
evidences:
  - "2"
self_check: I inspected the external JSON and log and asserted response marker2,
  matching focus/lens/web, six hover requests, absence of expensive endpoint
  calls and empty JavaScript errors. Parent cold-browser receipt separately
  covers post-deployment fresh selections.
---

# Browser addendum to the independent f22f223 review

Candidate f22f2239d489acaad42754c8e0955d5e188ea4a9. Reviewer dev-review@starter, 2026-10-26.

The browser coverage limit in evidence/review-f22f223-verdict.md is now resolved. The parent executed the reviewer's unchanged evidence/review_f22f223_browser.py outside the sandbox on its bounded owned Firefox driver at port4448. The reviewer inspected and asserted the resulting JSON and log after completion. This credits parent execution of a reviewer-authored test, not execution from the reviewer sandbox.

- Delayed A-to-B-to-A test passed. The old A connection response was deliberately delayed by three seconds while ignoring AbortController. The UI retained response marker2 after the old response arrived. Focus, selected map neuron, lens and web all remained L15 N417. The map retained48 connections.
- Actual pointer movement triggered six /api/hover requests. The captured network calls contained no /api/connections, /api/lens or /api/web requests; the only additional call was /api/status. No JavaScript errors were recorded. Combined with the independent HTTP check that uncached hover leaves database rows unchanged, this closes cache-only hover coverage.
- Receipt: evidence/review-f22f223-browser.json. Execution log: evidence/review-f22f223-browser-external.log. Screenshot: evidence/review-f22f223-browser.png. Original sandbox failure was retained by the parent as evidence/review-f22f223-browser-sandbox.json and .png; geckodriver failure log remains evidence/review-f22f223-geckodriver.log.

The candidate's scoped PASS stands with no requested product correction. Full-inference baseline logits were still not rerun; that numerical coverage limit remains and the inference admission gate must not be bypassed. Parent/operator still owns final service-control receipt, documentation and authorized publication.
