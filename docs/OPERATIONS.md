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

## Checks and research records

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
