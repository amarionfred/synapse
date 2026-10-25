# What can we uncover in open weights?

Park's question: can we expose useful capabilities that are present in a model but
poorly expressed, by understanding and changing its internal computation?

Synapse is the workspace for investigating this question together. Enthusiasm is
the motivation; held-out performance and causal evidence decide what worked.

## What Lyra's explorer does

The original [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)
hooks the input to Qwen3.5's MLP output projection. For a gated MLP,
`h = SiLU(W_gate x) * (W_up x)` and `y = W_down h`. Adding `alpha` to coordinate
`h[n]` changes the projection output by `alpha W_down[:,n]`. Lyra expresses the
additive dose relative to a layer's median residual norm on a reference text.

The two starting posts were [the demonstration](https://x.com/_lyraaaa_/status/2114211024962498831)
and [the stated recipe](https://x.com/_lyraaaa_/status/2113833330429083662).
The reported recipe uses Qwen3.5-4B L14 N8949 at +1 and L13 N7956 at -1.
Those are author-reported coordinates, not a cross-model recipe or a reproduced
capability gain. The local PLAYBOOK.md preserves the fuller initial investigation.

Synapse generalizes the intervention point to dense output projections. It reads
and changes activations during a forward pass. It does **not** modify saved model
weights. Multiplying a neuron's activation can be represented by scaling an
output column in a suitable floating-point checkpoint. An additive activation
shift generally needs a bias or an explicit injection; it is not the same edit.
Post-MLP normalization can also change the residual effect of a projection edit.

## A first capability experiment

1. Choose one mechanically graded task, such as small program execution or
   distractor-resistant arithmetic. Save train/search/held-out prompt sets before
   examining interventions. Use the same checkpoint, template and decoding policy.
2. Establish baseline accuracy and repeatability. Keep the individual outputs;
   a scalar average hides regressions.
3. Find candidate coordinates on the search split using activations or a
   contrastive direction. Compare +dose, -dose, zero/identity and norm-matched
   random interventions. Tune dose only on the search split.
4. Freeze the intervention and evaluate held-out paired accuracy. Measure
   unrelated tasks, verbosity, malformed answers, latency and memory too.
5. If it improves, test necessity, sufficiency and mediation separately. Reproduce
   across paraphrases and at least another checkpoint before making broad claims.

A change is evidence of useful capability access only if it improves the target
metric beyond controls without unacceptable regressions. Different wording,
confidence, compliance, or an attractive visualization is insufficient evidence.

## Candidate directions, not claims of novelty

| Experiment | Interesting mechanism | What could disprove it |
|---|---|---|
| Context-dependent neuron gains | A feature helps only at a reasoning stage or token position | Global scaling works equally well, or gains disappear on held-out tasks |
| Multi-neuron interactions | Two weak interventions cooperate or interfere | Their combined effect is explained by their individual effects |
| Contrastive activation steering | A correct-vs-incorrect direction changes computation | Random norm-matched directions give comparable gains |
| Causal path tests | An upstream feature affects output through a specific downstream feature | Ablating that mediator leaves the effect unchanged |
| Quantization sensitivity | Some write directions or features suffer disproportionately at low precision | Differences vanish after matching templates, calibration and numerical scales |
| Low-rank weight edits | A small persistent change preserves a useful intervention | Local success fails on paraphrases or damages unrelated behavior |
| Feature transfer across models | Matched representations support related interventions | Alignment fails to predict transfer beyond random matches |
| Conditional routing of existing features | Select when to apply an intervention using a small controller | The controller learns test leakage or adds more compute than the improvement warrants |

These have substantial prior art. Search primary literature before calling a
result new. Starting references: [ActAdd](https://arxiv.org/abs/2308.10248),
[ROME](https://rome.baulab.info/), [task arithmetic](https://arxiv.org/abs/2212.04089),
and [feature decomposition](https://www.anthropic.com/research/mapping-mind-language-model).
These are leads; they do not establish that the methods transfer unchanged to
Qwen3.5 or a quantized runtime.

## Interpretability and recursive improvement

This tool gives observational access and controlled interventions. Interpretability
also needs explanations that predict behavior in new settings, account for
distributed features and survive causal tests. A neuron label or graph does not
solve that problem.

Recursive self-improvement is a separate, much stronger claim: an improved system
must become better at producing subsequent improvements, with independently
checked gains and accounted costs across iterations. An agent searching edits is
an automated experiment loop; that alone demonstrates neither recursion nor an
intelligence explosion.

## Records

Use `.synapse/experiments/<date>-<question>/` for local experiment artifacts. Include
checkpoint ID/revision or weight hashes, adapter, precision, prompts and splits,
seeds, token positions, doses, baseline and control outputs, scoring code,
resource measurements, and a short claim/limitation statement. Agent-written
implementation and Park's predictions or manual observations should be identified.

Prediction exercise: before a probe, write whether increasing one chosen neuron's
activation should change a specific next token, an entire semantic feature, or
neither. Compare that prediction against the baseline and negative-dose result.
