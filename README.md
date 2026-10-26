# Synapse

A workspace for exploring open model internals with a human and a coding agent.
Forked from [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).
It retains the 3D neuron map, per-token activations, logit lens and paired steering,
adds **Codex CLI alongside Claude Code**, and replaces the Qwen-only loader with
validated dense-decoder adapters.

## Start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. The default lock uses
CPU PyTorch; choose an appropriate PyTorch build for a GPU installation.

```bash
uv sync --extra dev
python scripts/synapse.py serve --model HuggingFaceTB/SmolLM2-135M-Instruct
```

Open **http://127.0.0.1:8791**. Click **synapse → Load model for live experiments**,
or select a neuron and request an activation or steering probe. Loading happens
on demand. No full sweep or corpus indexing runs on startup.

The small SmolLM2 checkpoint makes live exploration practical on a CPU laptop.
It is a working research/demo model, not a substitute for a strong coding model.
Model downloads require disk space; CPU admission estimates check available RAM.

On Park's installed machine, the `synapse` command is on PATH:

```bash
synapse start              # bounded user service; 3 GiB (Transformers) / 5 GiB (GGUF), no swap
synapse status             # checkpoint, actual loaded state and capabilities
synapse codex              # interactive Codex with the same explorer MCP tools
synapse stop
```

`start` uses a conservative 3 GiB process limit for Transformers and 5 GiB for GGUF. For a larger
checkpoint on suitable hardware, use `serve` with an appropriate resource limit.
Select another model using `--model <Hugging-Face-ID-or-local-folder-or-GGUF-file>` and optionally
`--revision <commit>`. Stop the existing process first. Selection is remembered
locally; the **synapse** dialog generates the launch command.

For the installed **Qwen3.5-4B Q4_K_M** file, use the native inspection bridge:

```bash
python scripts/build_gguf_bridge.py  # C++ compiler + the installed llama.cpp b11669 runtime
qwen-lab stop
synapse stop
synapse start --model /absolute/path/to/Qwen3.5-4B-Q4_K_M.gguf
```

The bridge reads the GGUF's own tokenizer, quantized weights and live `ffn_swiglu`
tensors. It supports per-token activations, add/multiply interventions, paired
greedy generation and an on-demand logit lens. Weights stay quantized and unchanged
on disk. Qwen3.5 is the currently validated GGUF graph; other GGUF architectures
are rejected explicitly. The model name remains visible above the map. Editing
the launch-command preview does not switch the running server's checkpoint.

This CPU backend uses a 2,048-token context and generates the three steering rows
sequentially. Loading needs at least 4.5 GiB available RAM (more for larger files).
Synapse and `qwen-lab` share a process lock so they cannot load two Qwen contexts
at once. Stop Synapse before returning to `qwen-lab chat`. Calibration, neuron
names and caches belong to the quantized file; the author's BF16 data is separate.

For Lyra's shipped Qwen visualization without weights:

```bash
synapse start --author-map
```

That bundle is an author-provided exhibit. Its calibration and indexes are never
used for live inference or another checkpoint. Generated data, names and calibration
are stored separately by checkpoint/revision under `.synapse/models/`.

## Work with Codex

Install and authenticate [Codex CLI](https://learn.chatgpt.com/docs/non-interactive-mode)
once. Open the **Agent** tab and choose **Codex CLI**. The optional model field
selects the agent model; it is separate from the local checkpoint being examined.
The CLI default is used when blank. Claude Code remains available when installed.

Codex gets the explorer's MCP tools, the selected-neuron context, and this workspace.
It can inspect activations, run paired interventions, select/highlight neurons,
and write experiment scripts and notes. Tool calls and results appear in the panel.
Sessions resume across browser/server restarts; a new provider, agent model or
checkpoint gets separate conversation state. Codex edit/reroll is intentionally
not exposed as destructive session-file surgery; use **new** for a fresh thread.

The bridge invokes `codex exec --json`, keeps authentication in the existing CLI,
and uses a workspace-write sandbox. Other globally configured MCP servers are
disabled for that invocation. The known Synapse research/UI tools are explicitly allowed; shell access remains workspace-scoped. It does not copy credentials into the repository.
`.codex/config.toml` also makes the MCP available when this project is trusted;
`synapse codex` supplies its absolute path explicitly.

A newly opened agent thread reads the saved project context. It does not inherit
some other chat's complete conversation or become that already-running session.

## Compatibility

The adapters inspect module structure, projection orientation and dimensions.
Unsupported shapes produce an explicit error. A model-family name alone is not
proof that every variant, quantizer or operation works.

| Backend / architecture | Current scope |
|---|---|
| Dense Transformers: Llama, Mistral, Qwen2/3, Qwen3.5 hybrid, Gemma2, GPT-2, GPT-NeoX, OPT, Phi3 layouts | Activation capture, add/multiply interventions, three-way generation; small actual architecture tests |
| Final norm + unembedding | Logit lens when the adapter exposes both |
| Separate gate/up MLP + supported pre-MLP RMSNorm | Direct-path weight heuristic; explicitly reported as a heuristic |
| Compatible new dense variants | Structural detection, then runtime validation; not a blanket guarantee |
| Transformers MoE, encoder-decoder, nonuniform MLP widths, packed/quantized projections | Need additional adapters/backends; rejected explicitly |
| Qwen3.5 GGUF / pinned llama.cpp b11669 CPU | Live activations, add/multiply steering, three-way generation, per-neuron lens; validated on the installed 4B Q4_K_M file |
| Other GGUF architectures; GGUF head ablation, connection heuristics and full lens indexing | Not yet implemented; capability reporting makes this explicit |
| Closed API models | Can be research agents; weights and internal neurons are not available |

The hook coordinate is the input to the MLP output projection, after gating or
activation. Additive doses scale the projection output by a reference residual
norm. Post-MLP normalization can alter the eventual residual effect. Heads are
exposed only when their projection layout is validated. See `/api/agent/overview`.

Indexes for a new checkpoint are optional and are never borrowed from Qwen:

```bash
synapse stop
synapse index --scale --lens    # explicit work; can be costly for large vocabularies
synapse start
```

First-use activation calibration uses a short fixed reference text, clearly
labelled as such. It is not a corpus percentile. Full activation capture is bounded;
longer text is limited to the token budget returned by the server. Connection and layout builders now use the selected adapter and checkpoint cache; connection scoring requires a supported gated MLP. UMAP layouts require `uv sync --extra analysis`. Upstream vLLM GPU sweep scripts remain Qwen research utilities, not universal pipelines.

## Research and verification

Read [the research agenda](docs/RESEARCH_AGENDA.md) for experiments, controls and
what would count as evidence of untapped capability. Style changes, neuron labels
and colorful graphs do not demonstrate increased intelligence, solved
interpretability or recursive self-improvement.

```bash
EXPLORER_NO_MODEL=1 EXPLORER_BUNDLED_CACHE=1 .venv/bin/python -m pytest -q tests
```

Tests construct small real Transformers architectures without downloading weights.
They check unchanged baseline outputs, the additive column equation, whole-MLP
ablation, zero-dose continuation equivalence, cleanup and checkpoint isolation.
They establish implementation behavior, not capability improvement in trained models.
Machine-specific receipts and smoke outputs live locally under `evidence/`.

`scripts/verify_gguf_native.py` is an explicit real-checkpoint test for the installed
Qwen file. Run it with Synapse and `qwen-lab` stopped, under a memory limit. It
compares plain llama.cpp against observation and zero-dose callbacks, checks state
reset and nonzero effects, and checks capture across multiple processing batches.
The native bridge uses headers pinned by commit and SHA256, not arbitrary system
headers. See [the callback implementation](https://github.com/ggml-org/llama.cpp/blob/620d2225c/ggml/src/ggml-backend.cpp)
and [Qwen's graph](https://github.com/ggml-org/llama.cpp/blob/620d2225c/src/models/qwen35.cpp).

Park's Qwen3.5-4B Q4_K_M and OpenCode installation remains available through
`qwen-lab chat`; its server has a separate 5 GiB available-RAM gate. The detailed
initial investigation is saved locally in `PLAYBOOK.md`, with setup decisions in
`DECISIONS.md`. Model files, chat histories and those machine records are ignored
by Git.

## Provenance

Upstream revision: `f79b2196e29aacbad89684938baac8d97b172fd1`.
The [original README](docs/UPSTREAM_README.md), research notes and author caches
are retained. No LICENSE file was present in the inspected upstream revision;
this fork does not assign a new license to the author's code or data.
