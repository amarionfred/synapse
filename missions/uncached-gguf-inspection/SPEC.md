---
id: SYN.99.0.1
mission: uncached-gguf-inspection
stage: wip
verified: 2026-10-26 against scaffold (rig scope create)
created: 2026-10-26
intent: "Make arbitrary uncached neuron selections inspect GGUF weights without loading inference, with correct selection-owned UI and independent verification."
depends_on: []
---

# Mission — Uncached GGUF inspection repair

## Intent

Make arbitrary uncached neuron selections show their own links and token labels without loading inference. Preserve baseline inference, model bytes, owner applications and the existing project records.

## Scope and authority

The complete user request, technical evidence, six acceptance requirements and hover correction are in [AUTHORITY.md](AUTHORITY.md). Editing, building, local service deployment and pushing this bounded repair are authorized. Kernel operator coordinates; native builder implements and owns integration, independent reviewer returns a consolidated verdict, and the external parent validates the delivered result. No additional launch or planning question is needed.

Use the lightweight mission/slice workflow with one independent review at the completed candidate. No broader research, redesign, dependency/model downloads or unrelated cleanup. Canonical project AGENTS.md, README, PLAYBOOK.md and DECISIONS.md remain authoritative.

## Slices

- [SYN.99.0.1.1](slices/01-weight-only-selection/SPEC.md): weight-only selection, cache-only hover and stale-response protection.

## Status

Repair independently reviewed, deployed and published as f22f2239d489acaad42754c8e0955d5e188ea4a9. The actual cold browser, delayed selection race and hover paths passed. Final operator delivery uses the existing queue. Fresh full-inference baseline comparison remains an explicit RAM-gated verification limit. See slice PROOF.md and evidence/lithos-uncached-repair-result.md.
