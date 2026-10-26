# Reversible residual projection

The native Qwen3.5 research path captures a selected layer's full residual vector
at the final token of the rendered user/assistant prefix. It can project the
input embedding and all attention, recurrent-attention and MLP output writes:

    y <- y - strength * u * dot(u, y)

Here `u` is a unit vector in the model's hidden dimension, not an MLP neuron or
a calibrated neuron-add dose. Strength is limited to [-1,1]. A positive strength
removes the component; a negative strength amplifies it. Reversing the vector's
sign leaves the operation unchanged. In exact linear arithmetic this implements
`W' = W - strength * u * (u.T @ W)` for each residual writer. Quantized inference
and an independently exported adapter still require numerical comparison.

The operation is request scoped. It does not change the base GGUF, future chats,
the tied output readout, or the layer-end residual directly. Every intended
writer must execute once per token or the request fails. A layer-end projection
alone is a different operation, especially at partial strengths.

## Build and validate

Build from the checkout with `uv run python scripts/build_gguf_bridge.py`.
If using the optional Linux user service, restart it from a separate terminal;
restarting from an agent hosted inside that service ends its session:

    uv run python scripts/synapse.py stop
    uv run python scripts/synapse.py start

For a foreground server, stop its `serve` process with Ctrl-C and rerun the
original launch command. See [Operations](OPERATIONS.md) for both launch modes.

Then, with the same checkpoint selected and ordinary memory admission intact:

    .venv/bin/python scripts/verify_residual_projection.py --output .synapse/experiments/residual-live-check-01

The output directory must be new. This bounded check uses a random direction
and proves infrastructure only. It saves raw requests/results, checks >64 input
tokens, all writer callbacks, zero strength, a removed component and baseline
restoration. It never loads a second model. If resource admission fails, retain
the failed record and retry in a new directory when memory is available.

## Research API

`POST /api/agent/residual` accepts:

    {
      "model_key": "the exact key from /api/agent/overview",
      "text": "A user prompt",
      "layer": 21,
      "max_new": 32,
      "projection": {"direction": ["hidden-dimensional numeric unit vector"], "strength": 0.25}
    }

Omit `projection` for a baseline capture. The vector placeholder above must be
replaced by actual numbers. Inputs are bounded to 512 rendered tokens and
outputs to 128 tokens per row. Projection requests return baseline/intervened
greedy continuations, next-token top probabilities, baseline-to-intervention KL,
selected residual vectors, exact input/output token IDs, writer coverage and
perturbation energy. They also rerun the baseline prefill and check recovery.
Token positions refer to the rendered input including the assistant prefix.

`residual_research` in overview distinguishes implementation availability from
callback validation. Neither indicates that a useful refusal direction exists.
The default neuron-chat controls do not enable this research intervention.

## Direction and factor artifacts

`explorer/residual_projection.py` provides:

- `contrastive_direction(refused, answered)`: normalized difference of means
  from at least three matched, baseline-confirmed development examples.
- `export_factors(bridge, direction, destination, provenance)`: streamed block
  writer factors with shared `B=-u` and per-matrix `A=u.T @ W`. The checkpoint key
  and direction source are required. Existing artifacts cannot be overwritten.

The resulting NPZ is a Synapse callback adapter, **not a llama.cpp LoRA**.
It deliberately excludes the tied input embedding. Synapse applies the block
corrections to each writer's actual input and projects the input embedding
separately, leaving the tied readout unchanged. It requires the pinned graph's
plain matrix-multiply output nodes; unsupported shapes/operations fail.

`POST /api/agent/residual/factors` accepts `model_key`, `direction`,
`direction_source` (the experiment record), and a new `.npz` `destination` under
`.synapse/experiments/`. It streams weights and exports the factors without
enabling them. To test the result, replace `direction` in the residual request
with `artifact`, containing that file's project-relative or absolute path.
The checkpoint identity, target shapes and bounded archive sizes are validated.

The adapter adds `-strength*u*(A @ h)` to the existing quantized output. Direct
projection instead subtracts `strength*u*dot(u,y)` from that output. Quantized
input/matmul error can make them differ, so the integration check compares both.
Zero strength is an identity and each request checks subsequent baseline logits.
No merged-GGUF exporter is provided. Exporting a file does not enable an edit.

Before fitting, record prompts, splits and scoring rules. Only development data
may select the direction/strength. Separate refusal from inability, measure task
behavior rather than refusal vocabulary, compare positive/negative strengths
and random controls, and retain repetition/capability failures. All results and
limitations belong under `.synapse/experiments/` with exact checkpoint identity.
