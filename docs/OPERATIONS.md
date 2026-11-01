# Running Synapse

The commands below run from the repository root. They use the checked-in
[scripts/synapse.py](../scripts/synapse.py) launcher; no global shell wrapper is
required. The browser UI and HTTP/MCP tools share one selected checkpoint.

## Foreground server and model selection

```bash
uv sync
uv run python scripts/synapse.py serve --model HuggingFaceTB/SmolLM2-135M-Instruct
```

Open http://127.0.0.1:8791. Stop a foreground server with Ctrl-C before selecting
another checkpoint. Supply a Hugging Face ID, local Transformers directory, or
supported GGUF file with `--model`; `--revision` can pin a Transformers commit.
Use `--port` to change the port. For example:

```bash
uv run python scripts/synapse.py serve --model ./models/my-transformers-checkpoint
```

Selection is saved in `.synapse/settings.json`. Omitting `--model` reuses that
selection. The UI's model dialog generates a launch command; editing its preview
does not switch the running process. Loading weights is an explicit UI action or
happens on the first live probe. Startup does not run a sweep or build an index.

The default dependency configuration selects CPU PyTorch. A GPU deployment needs
an appropriate PyTorch installation and its own runtime validation. The browser
needs WebGL2 and access to the Three.js modules hosted on jsDelivr; initial model
downloads also need network access.

## Optional Linux user service

On Linux with a working systemd user session:

```bash
uv run python scripts/synapse.py start --model HuggingFaceTB/SmolLM2-135M-Instruct
uv run python scripts/synapse.py status
uv run python scripts/synapse.py stop
```

`start` runs `synapse-explorer.service` with a 3 GiB memory limit for Transformers
or 5 GiB for GGUF, no swap, and a CPU quota equivalent to three cores. These are
launcher defaults, not sufficient-memory promises for arbitrary models. A larger
model needs suitable hardware and a separately bounded foreground deployment.
`status` queries the HTTP server's identity, capabilities and loaded state; it
also works for a foreground server.

Stop the existing server before starting a replacement. When using the service,
restart it from a separate terminal if an agent is running inside it: stopping
the service also ends that agent turn.

## Codex, Claude and MCP

Install and authenticate the desired CLI separately, and make `codex` or
`claude` available on PATH. In **Agent**, choose the provider and optionally its
agent model. A blank model field uses the CLI default. This choice is independent
of the local checkpoint. Start a new conversation when switching provider,
agent model or checkpoint.

The web panel launches the CLI with the repository's
[stdio MCP bridge](../explorer/agent_mcp.py). Tools include overview, neuron and
connection inspection, activation traces, paired steering/log-probabilities,
prompt comparisons, and browser selection/highlighting. Availability depends on
the selected backend. Tools backed by sweep data return no sweep result when
none exists. The residual research endpoints are HTTP APIs, not dedicated tools
in this MCP list.

Codex can also be opened in a terminal with the same explorer MCP connection:

```bash
uv run python scripts/synapse.py codex
```

The launcher preserves the CLI's authentication. Its Codex invocation uses a
workspace-write sandbox and configures the Synapse tools explicitly; it disables
unrelated globally configured MCP servers for that invocation. Conversations
resume using saved session state. A new thread does not inherit another thread's
complete conversation.

## Native Qwen3.5 GGUF

This backend targets Linux CPU inference with the pinned llama.cpp b11669 runtime
(commit `620d2225c`) and a C++17 compiler. It has been exercised with
Qwen3.5-4B Q4_K_M, not all GGUF architectures or quantizations.

The existing [asset installer](../scripts/install_assets.py) downloads and verifies
that Qwen GGUF (about 2.74 GB), the Ubuntu x64 llama.cpp runtime, **and an OpenCode
binary**. OpenCode is not needed for Synapse's Codex/Claude panel. This is a
separate, optional asset bundle; `uv sync` does not download it. Inspect the
installer's pinned assets and disk requirements before choosing this path.

For that Ubuntu x64 bundle, with sufficient disk space:

```bash
mkdir -p evidence
uv run python scripts/install_assets.py
uv run python scripts/build_gguf_bridge.py
uv run python scripts/synapse.py serve --model ./models/Qwen3.5-4B-Q4_K_M.gguf
```

The build verifies the installed runtime version and downloads hash-pinned headers
if absent. Rebuild the bridge and restart the server after changing native code.
Alternatively, supply the expected pinned runtime layout and a compatible GGUF
yourself; the paths and checks are in [the builder](../scripts/build_gguf_bridge.py).

Links and per-neuron logit-lens labels use read-only weight inspection without an
inference context. Missing values are computed on selection; hover only reads
cached values. Multi-hop GGUF exploration allows one to three hops, limits a
request to 12 source expansions, and reports truncation.

Activation and generation probes require inference. Admission requires at least
4.25 GiB available RAM after Python starts, or the GGUF size plus 1.5 GiB,
whichever is larger. The native context is 2,048 tokens and three-way steering
runs sequentially on CPU. A shared inference lease prevents a second context
from the repository's separate Qwen chat runtime. These constraints do not apply
to the weight-only selection path.

## Model identity, calibration and indexes

Names, calibration and generated caches are stored by checkpoint identity under
`.synapse/models/`. Remote revisions and local file metadata distinguish model
selections. Data from Lyra's bundled Qwen map is never substituted for a live
checkpoint's calibration or indexes.

To view that cache-only exhibit without loading weights:

```bash
uv run python scripts/synapse.py serve --author-map
```

The hook coordinate is the input to the MLP output projection, after activation
or gating. Additive dose calibration uses a reference residual norm. First-use
activation scales come from a short fixed reference text, not corpus percentiles;
later normalization can change the residual effect. Read the server's reported
capabilities and calibration before comparing interventions.

A compatible Transformers checkpoint can explicitly build activation scales and
a searchable lens index. Stop its server first and run:

```bash
uv run python scripts/synapse.py index --model ./models/my-transformers-checkpoint --scale --lens
```

This loads the model and can be expensive. It is optional; a whole-model GGUF lens
index is not implemented. UMAP analysis dependencies are available with
`uv sync --extra analysis`. The inherited vLLM sweep scripts remain specialized
Qwen research utilities, not a general indexing pipeline.

### Map selection and connection lines

Selecting a neuron requests connection coordinates before unit metadata and
neighbor labels. Cached coordinates bypass the serial native inspection queue;
opening **Details** fills neighbor labels separately. The header reports link
loading and then its connection count. New, uncached coordinates still require
a read-only weight scan. Changing selection cancels stale GGUF label work between
bounded output-row chunks. The map keeps a selected neuron in the visible gap
between side panels, preserving zoom for clicks.

`node tests/map_selection_ui.cjs` checks geometry-before-label scheduling,
stale A→B→A responses, lazy labels and panel occlusion. These checks also run with
`scripts/verify_research_workflow.py --ui`.

## Bounded research requests

Read `GET /api/research/overview` first. It reports checkpoint identity,
calibration, implemented operations and live-validation status without loading
inference. Implementation availability does not establish real-checkpoint
numerical correctness. The existing memory admission, one-model lease and
no-swap service limits still apply. An unavailable operation returns an explicit
error; the server does not download or launch another model to satisfy it.

Complete-answer scoring uses `POST /api/research/score`:

```json
{
  "text": "Return the exact answer.",
  "completion": " 12",
  "chat": true,
  "prefix": "",
  "neurons": [],
  "eos_policy": "exclude",
  "timeout_seconds": 60
}
```

This requests one arm and one exact completion. Retain separate baseline and
intervention results for a paired comparison. Every contextual completion token
must be scored, with token IDs, per-token and total log probabilities retained.
An unstable prompt/completion token boundary is an error. `eos_policy` declares
whether an end-of-sequence token is excluded or included. Sequence likelihood
and exact parsed generation accuracy are different metrics. The legacy
`/api/agent/logprobs` endpoint remains a next-token probe; its `first_token_only`
warning must not be treated as a complete multi-token answer score.

Single-arm greedy generation uses `POST /api/research/generate`:

```json
{
  "messages": [{"role": "user", "content": "Reply with one digit."}],
  "neurons": [],
  "max_new": 32,
  "timeout_seconds": 60
}
```

Interventions use the existing `{ "layer": 0, "n": 0, "dose": 0.5,
"mode": "add" }` form; an empty list requests the baseline. These endpoints
accept at most 32 neuron interventions, 32 chat turns, a 256 KiB JSON body,
256 generated tokens and a finite timeout of at most 300 seconds. Runtime
context bounds can be smaller than the character limit. Unsupported fields and
head interventions are rejected. A request deadline or disconnect signals the
worker to stop; the serial slot remains held until the worker has actually
finished cleanup. Opening the UI or reading overview does not launch a study.

The stdio MCP bridge exposes `research_overview`, `score_sequence`,
`generate_one`, `capture_selected`, `calibrate_research`, `research_job`,
`intervention_preset` and `local_researcher`. The Codex panel includes these
named tools in its Synapse-only allowlist; its existing approval mode and user
disables remain intact. HTTP access uses the existing same-machine restriction.

## Paired jobs, presets and the local researcher

Open **Research** in the app to run a direct probe, submit a one-task paired job,
choose from a fixed action menu, or save/replay/compare/compose presets. Recent
paired jobs retain cancel/resume/status controls after a page reload. The HTTP API
and MCP also support larger, explicitly bounded item lists. Jobs, trial outcomes,
immutable presets and researcher episodes share `research.sqlite3` under the
checkpoint's existing state directory. A candidate does not create a new folder.

For additive research, explicitly calibrate the required layers before freezing
an experiment, preset or researcher episode:

```json
{"layers": [0], "timeout_seconds": 120}
```

Send this to `POST /api/research/calibrate`. Calibration uses the short fixed
reference, reports its forwards/token cost, and changes the calibration identity.
Multiplicative edits need no residual-norm calibration. A measured additive call
refuses a missing calibration instead of creating uncharged calibration work.
GGUF reuses its checkpoint-separated residual-norm manifest when present.

Submit an evaluator job to `POST /api/research/jobs`:

```json
{
  "name": "Bounded paired control",
  "mode": "generation", "split": "development", "seed": 0,
  "items": [{"id": "task-1", "prompt": "Return 7 + 5 as a number.", "answer": "12"}],
  "baseline": {"pushes": []},
  "treatment": {"pushes": [[0, 3, 0.9, "mul"]]},
  "max_new": 8,
  "limits": {"max_requests": 2, "max_tokens": 4096, "max_wall_seconds": 120}
}
```

Generation receives prompts without evaluator labels. Its exact-answer grader
ignores surrounding whitespace only. `mode: "sequence_score"` instead scores
the entire contextual answer; likelihood is not accuracy. Each job contains one
split: `development`, `validation` or `final`. Final requires an explicit
`candidate_frozen: true`; dataset custody remains the experimenter's responsibility.
Only `GET /api/research/jobs/{id}/feedback` on development jobs returns proposer
feedback. Validation/final feedback requests fail.

Use `GET /api/research/jobs` and `GET /api/research/jobs/{id}` to inspect jobs;
POST `{}` to `/{id}/cancel` or `/{id}/resume`. Resume runs only missing arms,
preserving spent budgets, successful baselines and unknown-outcome reservations.
After process interruption, jobs require explicit resume. The immutable spec and
checkpoint/backend/calibration identity must match. Request-token budgets charge
prompt plus generated/scored tokens; evaluated tokens and forwards are reported
separately. Before a call, a whole-context token reservation must fit. Unknown
work retains that reservation. Deadlines/cancellation are cooperative between
forwards; an already running native forward must finish before cleanup/slot reuse.

The CLI uses the existing server, never a second runtime:

```bash
.venv/bin/python scripts/run_experiment.py validate spec.json
.venv/bin/python scripts/run_experiment.py submit spec.json --wait 30
.venv/bin/python scripts/run_experiment.py status JOB_ID
.venv/bin/python scripts/run_experiment.py cancel JOB_ID
.venv/bin/python scripts/run_experiment.py resume JOB_ID --wait 30
```

`POST /api/research/presets` accepts `neurons`, optional `parents`, `label` and
`evaluation` provenance. GET the collection, `/{id}` or `/{id}/ancestry`. POST
`{left,right}` to `/compare`, `{ids:[...]}` to `/compose`, or `{id}` to `/replay`
and `/restore`. Restore with `id:null` returns the baseline. Replay returns
request-scoped pushes; it does not install persistent hooks or change weights.
Conflicting edits and incompatible identities fail explicitly. Stored evaluations
are caller-supplied provenance, not proof of a reproduced capability gain.

Create a measured proposer through `POST /api/research/researchers`:

```json
{
  "brief": "Choose the next intervention to test on the same target.", "seed": 0,
  "menu": [
    {"id": "control", "description": "Leave the target unchanged", "pushes": []},
    {"id": "candidate", "description": "Scale a declared MLP coordinate", "pushes": [[0,3,0.9,"mul"]]}
  ],
  "researcher_config": {"pushes": []}, "target_config": {"pushes": []},
  "budget": {"max_requests": 8, "max_tokens": 16384, "max_seconds": 120, "max_decisions": 4}
}
```

POST `{"feedback":[],"timeout_seconds":120}` to
`/api/research/researchers/{id}/choose`; GET `/{id}` for its durable journal.
The inspected local model scores **all tokens** of each displayed action ID.
Display order/IDs are permuted reproducibly; the menu, brief, seed and both
configurations freeze when the episode is created. Feedback accepts only
`{split:"development",action_id,correct,total}` summaries for that menu.
Researcher budgets charge evaluated tokens, including prompt recomputation.
Reservations commit before native calls; an interrupted request stays charged.
Episode wall time includes idle/interrupted time. No automatic inference runs
on restart. Choosing an action does not evaluate its target intervention: submit
that action separately to the evaluator, keeping original/edited proposers on
identical starting targets and budgets. The external Agent must not adapt the
measured prompt/menu between decisions.

## Selected activation capture and scoped edits

`POST /api/research/capture` accepts `text`, `chat`, `prefix`, optional `neurons`
and `selection: [[layer,position,neuron], ...]` (at most 4096 scalars). It captures
MLP output-projection inputs before edits and returns a hashed reference.
Score/generate accept `interventions` entries with `layer`, `n`, `dose`, `mode`,
`prompt_positions` and `decode_steps`. Omitted masks select that whole phase;
`[]` selects none. Positions refer to the rendered/tokenized prompt.
`decode_steps[0]` is the first generated token fed back to predict the second
output token. GGUF positions remain absolute across 64-token prefill chunks and
cached decode. Dense backends recompute the full prefix under the same edits.

For bounded clean/corrupt patching, pass
`patches: {reference: CAPTURE_RESPONSE, target_token_ids: [...]}` to score/generate.
Reference and target must have equal prompt token counts; the target IDs must
match the declared target exactly. Capture hashes and checkpoint/backend/
calibration identity are checked. Patching conflicts with steering on the same
coordinate. Every request starts fresh; cached prefixes are never reused across
changed interventions. This path does not assume standard attention layouts or
implement GGUF attention-head editing.

## Checks and research records

For serialized offline verification, including HTTP/client contracts and native
CPU tensor fixtures:

```bash
.venv/bin/python scripts/verify_research_workflow.py --ui --native
```

On Linux with a user systemd session, this runs under a 1 GiB memory cap, zero
swap and one CPU, and reports the cgroup peak/oom counters. `--ui` requires Node;
`--native` requires already installed pinned headers/runtime and a C++ compiler.
No checkpoint inference or downloads occur. Client checks use synthetic DOM/HTTP
fixtures; native checks use actual GGML CPU buffers. These do not establish live
Qwen numerical equivalence or capability gains.


```bash
uv sync --extra dev
EXPLORER_NO_MODEL=1 EXPLORER_BUNDLED_CACHE=1 uv run python -m pytest -q tests
```

The architecture tests construct small Transformers models without downloading
trained weights. They check capture, intervention math, baseline preservation,
cleanup and checkpoint isolation. They establish implementation behavior, not
capability gains.

Real-checkpoint checks are separate and resource-intensive: the
[native callback check](../scripts/verify_gguf_native.py),
[connection check](../scripts/verify_gguf_connections.py) and
[inspection-only check](../scripts/verify_gguf_inspection.py) have different scopes.
The inspection check's optional `--reference` mode requires admitted inference.
Do not load competing models to run them.

[Residual projection](RESIDUAL_PROJECTION.md) is experimental; live integration
validation is pending. Its factor archive is a separate callback adapter and does
not modify the base GGUF. Keep experiment prompts, controls, checkpoint identity,
outputs and limitations under `.synapse/experiments/`, following the
[research agenda](RESEARCH_AGENDA.md). Models, caches, credentials, conversations
and machine-specific records stay out of Git.
