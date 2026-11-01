"""Stdio MCP server that gives the Synapse's agent chat its tools: read any neuron data through the explorer's HTTP API,
run the live model (steered rollouts, activations) when it is loaded, and drive the map (select, highlight) in the browser that started the chat. Standard library only; started by
Codex CLI or Claude Code from server.py with EXPLORER_URL and EXPLORER_CHAT set.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
CHAT = os.environ.get("EXPLORER_CHAT", "")

LN = {"layer": {"type": "integer", "description": "zero-based layer; bounds from overview"}, "n": {"type": "integer", "description": "zero-based MLP output-projection input; bounds from overview"}}
RESEARCH_NEURONS = {"type": "array", "maxItems": 32, "items": {
    "type": "object", "properties": {**LN, "dose": {"type": "number", "default": 0.8},
                                        "mode": {"type": "string", "enum": ["add", "mul"]}},
    "required": ["layer", "n"], "additionalProperties": False}}
SCOPES = {
    "interventions": {"type": "array", "maxItems": 128, "items": {"type": "object",
        "properties": {**LN, "dose": {"type": "number"}, "mode": {"enum": ["add", "mul"]},
            "prompt_positions": {"type": "array", "maxItems": 2048, "items": {"type": "integer", "minimum": 0, "maximum": 2047}},
            "decode_steps": {"type": "array", "maxItems": 1024, "items": {"type": "integer", "minimum": 0, "maximum": 1023}}},
        "required": ["layer", "n", "dose", "mode"], "additionalProperties": False}},
    "patches": {"type": "object", "properties": {"reference": {"type": "object"},
        "target_token_ids": {"type": "array", "maxItems": 2048, "items": {"type": "integer"}}},
        "required": ["reference", "target_token_ids"], "additionalProperties": False},
}
TOOLS = [
    {"name": "overview", "description": "Model architecture and which layers have sweep runs (rollouts and rankings).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "research_overview", "description": "Research operation availability, checkpoint provenance, limits and live-validation status. Does not load inference. The inspected local model is separate from the external Agent.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "score_sequence", "description": "Score every token of one exact completion in rendered prompt context using teacher forcing. Returns contextual token IDs, per-token and total likelihood and provenance. This is a single arm; retain baseline/control results separately. Rejects unstable token boundaries. Live inference admission applies.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string", "minLength": 1, "maxLength": 32768},
         "completion": {"type": "string", "minLength": 1, "maxLength": 32768},
         "chat": {"type": "boolean", "default": True},
         "prefix": {"type": "string", "maxLength": 32768},
         **SCOPES, "neurons": RESEARCH_NEURONS,
         "eos_policy": {"type": "string", "enum": ["exclude", "include"], "default": "exclude"},
         "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 300, "default": 60}},
         "required": ["text", "completion"], "additionalProperties": False}},
    {"name": "generate_one", "description": "Generate one greedy arm from the inspected local model, with optional request-scoped MLP interventions. Returns exact output tokens, counts, stop reason and provenance; preserves baseline restoration and existing inference admission. Does not invoke the external Agent or launch a study.",
     "inputSchema": {"type": "object", "properties": {
         "messages": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
             "type": "object", "properties": {"role": {"type": "string", "enum": ["user", "assistant", "system"]},
                                                 "content": {"type": "string", "minLength": 1, "maxLength": 32768}},
             "required": ["role", "content"], "additionalProperties": False}},
         **SCOPES, "neurons": RESEARCH_NEURONS,
         "max_new": {"type": "integer", "minimum": 1, "maximum": 256, "default": 32},
         "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 300, "default": 60}},
         "required": ["messages"], "additionalProperties": False}},
    {"name": "capture_selected", "description": "Capture up to 4096 MLP scalars at [layer,position,neuron] before edits. Returns a checkpoint-bound hashed reference for matched activation patching; no full activation cube.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"},
       "selection": {"type": "array", "minItems": 1, "maxItems": 4096, "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer"}}},
       "chat": {"type": "boolean"}, "prefix": {"type": "string"}, "neurons": RESEARCH_NEURONS,
       "timeout_seconds": {"type": "number", "maximum": 300}}, "required": ["text", "selection"], "additionalProperties": False}},
    {"name": "calibrate_research", "description": "Explicit bounded normalization on the fixed reference, before freezing additive experiments or presets. Uses existing inference admission and returns its cost and updated calibration identity.",
     "inputSchema": {"type": "object", "properties": {"layers": {"type": "array", "minItems": 1, "maxItems": 32, "items": {"type": "integer"}}, "timeout_seconds": {"type": "number", "maximum": 300}}, "required": ["layers"], "additionalProperties": False}},
    {"name": "research_job", "description": "Submit/list/status/cancel/resume bounded paired evaluator jobs using the existing serial runtime and SQLite store. Spec declares one split, exact labeled items, baseline/treatment pushes and request/token/wall limits. Final requires candidate_frozen=true. Feedback returns aggregate development outcomes only.",
     "inputSchema": {"type": "object", "properties": {"action": {"enum": ["submit", "list", "status", "feedback", "cancel", "resume"]}, "id": {"type": "string"}, "spec": {"type": "object"}}, "required": ["action"], "additionalProperties": False}},
    {"name": "intervention_preset", "description": "Save/list/get/compare/compose/replay/restore immutable MLP presets and ancestry in the existing store. Identity includes checkpoint, backend and calibration. Replay/restore returns next-request pushes; no persistent hook or weight change. Save accepts neurons, parents, evaluation and label in payload; compare takes left/right, compose ids, replay/restore id.",
     "inputSchema": {"type": "object", "properties": {"action": {"enum": ["save", "list", "get", "ancestry", "compare", "compose", "replay", "restore"]}, "id": {"type": "string"}, "payload": {"type": "object"}}, "required": ["action"], "additionalProperties": False}},
    {"name": "local_researcher", "description": "Create/status/choose a durable episode where the inspected local model scores a frozen action menu. Create protocol has menu [{id,description,pushes}], brief, seed, researcher_config, target_config and budget {max_requests,max_tokens,max_seconds,max_decisions}. Choose accepts development aggregates [{split,action_id,correct,total}] only. The external Agent must not adapt the frozen brief/menu. Chosen actions are evaluated separately on the identical target.",
     "inputSchema": {"type": "object", "properties": {"action": {"enum": ["create", "status", "choose"]}, "id": {"type": "string"}, "protocol": {"type": "object"}, "feedback": {"type": "array"}, "timeout_seconds": {"type": "number", "maximum": 300}}, "required": ["action"], "additionalProperties": False}},
    {"name": "neuron", "description": "Inspect one neuron, computing supported live data on demand: name, logit lens and steer lens (with KL) for "
     "both poles, activation range, sweep rank if any, and its 24 strongest direct-path links each way (writes_into, "
     "reads_from) with each linked neuron's available lens labels, top words and name.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "rollouts", "description": "Sweep rollouts for a neuron: steered text per prompt and dose beside the "
     "unsteered baselines, distinctive words and stats. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {**LN, "max_chars": {"type": "integer", "default": 500}},
                     "required": ["layer", "n"]}},
    {"name": "cone", "description": "The neuron's connection cone: links chained forward (downstream) and backward "
     "(upstream), following top k links. GGUF computes/caches weights on demand with 1..3 hops and a reported expansion limit. Returns the neurons reached with hop, best z and "
     "lens labels. Multi-hop paths are weight-only guesses: leads to check, not circuits.",
     "inputSchema": {"type": "object", "properties": {**LN, "k": {"type": "integer", "default": 2},
                                                       "limit": {"type": "integer", "default": 120}, "hops": {"type":"integer","minimum":1,"maximum":3,"default":1}},
                     "required": ["layer", "n"]}},
    {"name": "web_summary", "description": "The token summary of the neuron's web, the same one the person sees in the "
     "left panel: lens tokens pooled over its cone, each connected neuron read on the pole the chain drives. The response states the ranking: indexed layerwise lift or on-demand weighted token counts. `toward` / `away`: what its downstream neurons push the model toward and away "
     "from when this neuron fires (indirect effect). `fed_by` / `held_by`: what its upstream writers say when they "
     "excite / inhibit it (its inputs). Each token has n (neurons carrying it) and a score.",
     "inputSchema": {"type": "object", "properties": {**LN, "first": {"type": "integer", "default": 5},
                                                       "k": {"type": "integer", "default": 2}, "hops": {"type":"integer","minimum":1,"maximum":3,"default":1}},
                     "required": ["layer", "n"]}},
    {"name": "search_lens", "description": "Neurons whose logit lens or steer lens (top 12, either pole) contains a token "
     "matching the query (case-insensitive substring). Best matches first.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 40}},
                     "required": ["query"]}},
    {"name": "top_units", "description": "A layer's neurons ranked by sweep word-cluster mass against random-direction "
     "controls, with top words and names. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {"layer": LN["layer"], "limit": {"type": "integer", "default": 30}},
                     "required": ["layer"]}},
    {"name": "set_name", "description": "Name a neuron in the explorer (the name field in its panel, shown in lists and on "
     "the map), or clear its name with an empty string. Only when the person asks, or after they agree to a name you "
     "suggested.",
     "inputSchema": {"type": "object", "properties": {**LN, "name": {"type": "string"}}, "required": ["layer", "n", "name"]}},
    {"name": "names", "description": "Every neuron the person has named so far, as {\"L14/N3262\": name}.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "steer", "description": "Live: generate from a chat prompt three ways in one batch, pushed up, unsteered, "
     "and pushed down (greedy). Steer one neuron (layer, n) or several at once (neurons: [{layer, n, dose?, mode?}]; all "
     "are applied together, each reversed in the down row; a negative per-neuron dose pushes that one the other way). "
     "dose in units of the layer's median residual norm for mode 'add' (0.8 is the steer-lens push; 2-3 is strong), or "
     "a multiplier for mode 'mul'. Needs the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "prompt": {"type": "string"},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer", "n"]}},
                                                       "dose": {"type": "number", "default": 0.8},
                                                       "mode": {"type": "string", "enum": ["add", "mul"], "default": "add"},
                                                       "max_new": {"type": "integer", "default": 150}},
                     "required": ["prompt"]}},
    {"name": "activations", "description": "Live: the neuron's raw activation on every token of a text (chat=true wraps "
     "it as a user turn). Compare against act_range from `neuron`. Optional `neurons` [{layer, n, dose?, mode?}] are "
     "pushed during the same pass (steering units), to test causally whether they drive this neuron: compare with the "
     "same call without them. {layer, all: true, dose: 0} zeroes that layer's whole MLP. Optional `heads` [{layer, "
     "head, dose}] scale attention heads (dose 0 = ablate; head 'all' = every head; head counts depend on the checkpoint; supported heads are in `overview`). Only earlier layers can move it. Needs "
     "the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"},
                                                       "chat": {"type": "boolean", "default": False},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["layer", "n", "text"]}},
    {"name": "compare", "description": "Live: read many neurons across several texts in one call (one forward pass per "
     "text). Per neuron and text: its strongest positive and negative activation with the token, and peak_score (the "
     "activation over the neuron's usual extreme; 1 = as strong as its usual max). For screening candidates against "
     "target texts and controls; `activations` gives one neuron's full per-token trace and takes pushes for causal "
     "tests. Up to 300 neurons and 24 texts.",
     "inputSchema": {"type": "object", "properties": {
         "neurons": {"type": "array", "items": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
         "texts": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {
             "label": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}]}},
         "chat": {"type": "boolean", "default": False}},
                     "required": ["neurons", "texts"]}},
    {"name": "logprobs", "description": "Live: the next token's log-probabilities after a text (chat=true by default: "
     "the text is a user turn and the next token opens the reply; `prefix` continues from the start of a reply or any "
     "text). Returns the top k tokens and the logprob of each `targets` string's first token. With `neurons` and/or "
     "`heads` (same as `activations`: pushes, whole-layer zeroing, head scaling) it runs with and without them and "
     "returns both, the KL from baseline to intervened, and the tokens that gained and lost the most probability "
     "across the whole vocabulary: a direct read of what an intervention does to the output.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "chat": {"type": "boolean", "default": True},
                                                       "prefix": {"type": "string"}, "k": {"type": "integer", "default": 15},
                                                       "targets": {"type": "array", "items": {"type": "string"}},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["text"]}},
    {"name": "top_firing", "description": "Live: run text through the model and list the neurons firing hardest at one "
     "token (the first token containing `token`; omit it for the token with the strongest firing). Each neuron has its "
     "pole, score (activation over its usual extreme; 1 = as strong as its usual max), raw activation and steer lens. "
     "`layers` [lo, hi] restricts the layers. Doesn't change the person's screen; use show_text to show them.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 30},
                                                       "layers": {"type": "array", "items": {"type": "integer"}},
                                                       "chat": {"type": "boolean", "default": False}},
                     "required": ["text"]}},
    {"name": "camera", "description": "Move the person's 3D view. mode 'home': the whole model. 'layer': look at one "
     "layer's neuron sheet face-on (give layer). 'neuron': fly close to one neuron (give layer and n; closeness 0.1 = very "
     "close, 0.5 = wide).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["home", "layer", "neuron"]},
                                                       **LN, "closeness": {"type": "number", "default": 0.22}},
                     "required": ["mode"]}},
    {"name": "annotate_neurons", "description": "Pin short text labels to neurons on the 3D map (they follow the camera). "
     "Replaces the previous labels; an empty list clears them. Keep each label a few words.",
     "inputSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "text": {"type": "string"}}, "required": ["layer", "n", "text"]}}},
                     "required": ["items"]}},
    {"name": "show_text", "description": "Put text in the explorer's paste box and run it through the live model, so the "
     "person sees its tokens; pin the first token containing `token` to show the neurons firing most there and the "
     "direct links among them on the map. Needs the live model.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "show_search", "description": "Run a lens search in the explorer's left panel, so the person sees the "
     "matching neurons listed and can click them.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "open_tab", "description": "Switch the open neuron's panel to a tab: examples (sweep rollouts), chat (the "
     "person's own steered chat) or details (lenses and connections).",
     "inputSchema": {"type": "object", "properties": {"tab": {"type": "string", "enum": ["examples", "chat", "details"]}},
                     "required": ["tab"]}},
    {"name": "select_neuron", "description": "Open a neuron in the explorer: its panel, and its connection web on the map.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "highlight_neurons", "description": "Mark a set of neurons on the 3D map (replaces the previous marks; an "
     "empty list clears them). sign '-' draws a neuron in the negative-pole colour.",
     "inputSchema": {"type": "object", "properties": {"neurons": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "sign": {"type": "string", "enum": ["+", "-"]}},
         "required": ["layer", "n"]}}}, "required": ["neurons"]}},
]


def get(path, **q):
    url = f"{URL}{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def post(path, body, timeout=60):
    req = urllib.request.Request(f"{URL}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def call(name, a):
    if name == "overview":
        return get("/api/agent/overview")
    if name == "research_overview":
        return get("/api/research/overview")
    if name == "score_sequence":
        return post("/api/research/score", a, timeout=900)
    if name == "generate_one":
        return post("/api/research/generate", a, timeout=900)
    if name in {"capture_selected", "calibrate_research"}:
        return post("/api/research/" + ("capture" if name == "capture_selected" else "calibrate"), a, timeout=900)
    if name == "research_job":
        action = a["action"]
        if action == "submit":
            return post("/api/research/jobs", a["spec"])
        if action == "list":
            return get("/api/research/jobs")
        path = "/api/research/jobs/" + urllib.parse.quote(a["id"], safe="")
        if action in {"cancel", "resume"}:
            return post(path + "/" + action, {})
        return get(path + ("/feedback" if action == "feedback" else ""))
    if name == "intervention_preset":
        action, payload = a["action"], a.get("payload", {})
        path = "/api/research/presets"
        if action == "list":
            return get(path)
        if action in {"get", "ancestry"}:
            return get(path + "/" + urllib.parse.quote(a["id"], safe="") + ("/ancestry" if action == "ancestry" else ""))
        return post(path + ("" if action == "save" else "/" + action), payload)
    if name == "local_researcher":
        action = a["action"]
        path = "/api/research/researchers"
        if action == "create":
            return post(path, a["protocol"])
        path += "/" + urllib.parse.quote(a["id"], safe="")
        if action == "status":
            return get(path)
        return post(path + "/choose", {"feedback": a.get("feedback", []), "timeout_seconds": a.get("timeout_seconds", 60)}, timeout=900)
    if name == "neuron":
        return get("/api/agent/neuron", layer=a["layer"], n=a["n"])
    if name == "rollouts":
        u = get("/api/unit", layer=a["layer"], n=a["n"])
        cut = int(a.get("max_chars", 500))
        for key in ("rollouts", "baselines"):
            for r in u.get(key, []):
                r["text"] = r["text"][:cut]
        return u
    if name == "cone":
        return get("/api/agent/cone", layer=a["layer"], n=a["n"], k=a.get("k", 2), limit=a.get("limit", 120), hops=a.get("hops",1))
    if name == "web_summary":
        r = get("/api/web/summary", layer=a["layer"], n=a["n"], first=a.get("first", 5), k=a.get("k", 2), top=20, hops=a.get("hops",1))
        slim = lambda lst: [f"{t['tok']} ({t['n']})" for t in lst]
        return {"lens": r["src"], "links": r["links"],
                "ranking":r.get("ranking","layerwise lift"),"hops":r.get("hops"),"truncated":r.get("truncated",False),
                "downstream_neurons": r["down"]["neurons"], "toward": slim(r["down"]["toward"]), "away": slim(r["down"]["away"]),
                "upstream_neurons": r["up"]["neurons"], "fed_by": slim(r["up"]["fed_by"]), "held_by": slim(r["up"]["held_by"])}
    if name == "search_lens":
        r = get("/api/search", q=a["query"], limit=a.get("limit", 40))
        return {"tokens_matched": r.get("tokens"), "total": r.get("total"), "results": r.get("results", [])}
    if name == "top_units":
        us = sorted((u for u in get("/api/units", layer=a["layer"]) if u["kind"] == "neuron"), key=lambda u: -u["z"])
        return us[: int(a.get("limit", 30))]
    if name == "set_name":
        post("/api/name", {"layer": a["layer"], "n": a["n"], "name": a["name"]})
        post("/api/agent/ui", {"chat": CHAT, "action": "rename", "layer": a["layer"], "n": a["n"], "name": a["name"]})
        return {"ok": True, "neuron": f"L{a['layer']} N{a['n']}", "name": a["name"].strip() or None}
    if name == "names":
        return get("/api/agent/names")
    if name == "steer":
        return post("/api/agent/steer", a, timeout=900)  # the first call may wait for the model to load
    if name == "top_firing":
        return post("/api/agent/fires", a, timeout=900)
    if name == "logprobs":
        return post("/api/agent/logprobs", a, timeout=900)
    if name == "compare":
        return post("/api/agent/compare", a, timeout=900)
    if name == "activations":
        return post("/api/agent/activations", a, timeout=900)
    ui = {"camera": "camera", "annotate_neurons": "annotate", "show_text": "text", "show_search": "search", "open_tab": "tab"}
    if name in ui:
        return post("/api/agent/ui", {"chat": CHAT, "action": ui[name], **a})
    if name == "select_neuron":
        return post("/api/agent/ui", {"chat": CHAT, "action": "select", "layer": a["layer"], "n": a["n"]})
    if name == "highlight_neurons":
        return post("/api/agent/ui", {"chat": CHAT, "action": "highlight", "neurons": a["neurons"]})
    raise ValueError(f"unknown tool {name}")


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    if not line.strip():
        continue
    m = json.loads(line)
    method, id_ = m.get("method"), m.get("id")
    if id_ is None:
        continue  # notifications
    if method == "initialize":
        reply(id_, {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {}}, "serverInfo": {"name": "explorer", "version": "0.1"}})
    elif method == "tools/list":
        reply(id_, {"tools": TOOLS})
    elif method == "tools/call":
        p = m["params"]
        try:
            out = call(p["name"], p.get("arguments") or {})
            reply(id_, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})
        except Exception as e:  # tool errors go back to the model, not the protocol
            reply(id_, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    elif method == "ping":
        reply(id_, {})
    else:
        reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
