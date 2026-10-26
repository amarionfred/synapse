# Synapse

This is Amarion's neuron exploration workspace, forked from Lyra's neuron-explorer.
The user's name is Amarion; `park` is only the existing system username/path.
Use the Synapse MCP `overview` tool before interpreting any neuron. The checkpoint,
revision, supported operations, calibration and loaded status come from the server.
Never apply Qwen's bundled data or neuron numbers to a different checkpoint.

Read README.md for operations and docs/RESEARCH_AGENDA.md for the research question.
Local PLAYBOOK.md and DECISIONS.md preserve the initial investigation and setup.
Record experiments under .synapse/experiments/ with checkpoint identity, prompts,
seeds, doses, controls, outputs and limitations. Research claims need actual results.
UI links and weight similarities are hypotheses, not demonstrated causal circuits.

Keep models, caches, credentials, chat histories and machine-specific records out of Git.
The machine's /tmp is RAM-backed; use the project or TMPDIR for large artifacts.
Do not load a second large model while the local Qwen service occupies memory.
Do not run broad sweeps, modify saved weights, or remove data as part of a simple probe.
Use reversible activation interventions for initial tests. Retain baselines and hashes.
This agent can edit this workspace and call its running explorer; a new Codex thread
does not automatically inherit the conversation that originally created the project.
