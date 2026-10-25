This is the working directory for local OpenCode trials. `probe` uses a short prompt and no tools, to make text comparisons inexpensive and repeatable. Select OpenCode's built-in `build` agent when you want coding and tool use. Keep the same agent, prompt, and tool settings across baseline and treatment comparisons.

The configured model is the unchanged Qwen3.5-4B Q4_K_M baseline. The neuron steering described in `../PLAYBOOK.md` is not active in this server.

For a bounded text-only baseline, supply a session title so the local model does not also generate one:

```sh
qwen-lab chat run --pure --agent probe --title "Baseline probe" "Reply with only the result of 17 + 25."
qwen-lab stop
```

OpenCode adds its own prompt around a question, so this can take substantially longer than a direct API call on a CPU. This is a plumbing check, not a capability benchmark.
