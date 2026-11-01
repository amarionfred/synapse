"""Complete contextual likelihood; EOS excluded by default, no length penalty.

Include appends one declared EOS. Unstable prompt/answer boundaries are rejected
instead of substituting separately tokenized answers. Mean is reported separately.
"""
import math
import time
import sys
from contextlib import contextmanager
import numpy as np
from interventions import Plan, digest, selection_rows


def exact_grade(output, answer):
    if not isinstance(output, str) or not isinstance(answer, str):
        raise ValueError("Exact grading requires string output and reference")
    return output == answer


def _text(value, name, allow_empty=True):
    if not isinstance(value, str) or len(value.encode()) > 65536 or (not allow_empty and not value):
        raise ValueError(f"{name} must be a bounded string")
    return value


def _deadline(deadline):
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite absolute monotonic time")


def interrupted(cancel, deadline):
    if cancel is not None and cancel.is_set():
        return "canceled"
    if deadline is not None and time.monotonic() >= deadline:
        return "deadline"
    return None


@contextmanager
def _lease(runtime, cancel, deadline):
    acquired = False
    try:
        while not acquired:
            reason = interrupted(cancel, deadline)
            if reason:
                yield reason
                return
            acquired = runtime.lock.acquire(timeout=.05)
        yield interrupted(cancel, deadline)
    finally:
        if acquired:
            runtime.lock.release()


def _not_started(runtime, reason, kind, eos_policy=None):
    result = dict(completed=False, finish_reason=reason, execution_started=False,
                  prompt_token_ids=[], counts=_counts(0), identity=runtime.research_identity(),
                  cleanup=dict(interventions_cleared=True, cache_cleared=True))
    if kind == "score":
        result.update(completion_token_ids=[], per_token_logprobs=[], total_logprob=None,
                      mean_logprob=None, eos_policy=eos_policy)
    elif kind == "generate":
        result.update(output_token_ids=[], text="", stop_token_id=None, eos_policy="stop-before-eog")
    return result


def contextual_tokens(runtime, text, completion, chat=True, prefix=""):
    _text(text, "text")
    _text(completion, "completion", False)
    _text(prefix, "prefix")
    if type(chat) is not bool:
        raise ValueError("chat must be boolean")
    rendered, specials = runtime._research_render(text, chat)
    rendered += prefix
    prompt_ids = list(runtime.tok(rendered, add_special_tokens=specials).input_ids)
    full_ids = list(runtime.tok(rendered + completion, add_special_tokens=specials).input_ids)
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Completion retokenizes the prompt boundary; choose a stable explicit delimiter")
    target = full_ids[len(prompt_ids):]
    if not target:
        raise ValueError("Completion has no contextual tokens")
    return prompt_ids, target


def _logprob(logits, token):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= token < len(values):
        raise ValueError("Invalid logits or target token")
    mx = float(values.max())
    return float(values[token] - mx - np.log(np.exp(values - mx).sum()))


def _counts(prompt_count, requested=0):
    return dict(prompt_tokens=prompt_count, output_tokens=0, requested_tokens=requested,
                scored_tokens=0, forward_passes=0, evaluated_tokens=0)


def _forward(runtime, full_ids, plan, reset, counts, capture=()):
    ids, start = runtime._research_input(full_ids, reset)
    edits = plan.edits(start, len(ids))
    counts["forward_passes"] += 1
    counts["evaluated_tokens"] += len(ids)
    return runtime._research_eval(ids, reset=reset, pushes=plan.global_pushes,
                                  edits=edits, capture=capture, position_offset=start)


def _failure(exc, counts, started):
    try:
        exc.research_costs = {**counts, "wall_seconds": time.monotonic() - started}
    except (AttributeError, TypeError):
        pass


def _cleanup(runtime):
    failure = sys.exception()
    runtime._research_clear()
    if failure is not None:
        failure.research_cleanup = dict(interventions_cleared=True, cache_cleared=True)


def score_sequence(runtime, text, completion, chat=True, prefix="", pushes=None,
                   eos_policy="exclude", cancel=None, deadline=None, *, interventions=None, patches=None):
    _deadline(deadline)
    if eos_policy not in {"exclude", "include"}:
        raise ValueError("eos_policy must be exclude or include")
    started = time.monotonic()
    with _lease(runtime, cancel, deadline) as stop:
        if stop:
            return _not_started(runtime, stop, "score", eos_policy)
        runtime.ensure_loaded()
        prompt, target = contextual_tokens(runtime, text, completion, chat, prefix)
        if eos_policy == "include":
            eos = runtime._research_eos()
            if eos is None:
                raise ValueError("This tokenizer has no declared EOS token")
            target.append(eos)
        if len(target) > 1024 or len(prompt) + len(target) > runtime._research_context_limit():
            raise ValueError("Prompt plus complete answer exceeds the bounded context")
        counts = _counts(len(prompt), len(target))
        scores, ids, reason = [], list(prompt), "completed"
        try:
            plan = Plan(runtime, prompt, pushes, interventions, patches)
            for i, token in enumerate(target):
                stop = interrupted(cancel, deadline)
                if stop:
                    reason = stop
                    break
                logits, _ = _forward(runtime, ids, plan, i == 0, counts)
                stop = interrupted(cancel, deadline)
                if stop:
                    reason = stop
                    break
                scores.append(_logprob(logits, token))
                counts["scored_tokens"] += 1
                ids.append(token)
        except Exception as exc:
            _failure(exc, counts, started)
            raise
        finally:
            _cleanup(runtime)
        counts["wall_seconds"] = time.monotonic() - started
        return dict(prompt_token_ids=prompt, completion_token_ids=target,
                    per_token_logprobs=scores, total_logprob=float(sum(scores)),
                    mean_logprob=float(sum(scores) / len(scores)) if scores else None,
                    completed=len(scores) == len(target), finish_reason=reason,
                    eos_policy=eos_policy, boundary_policy="contextual-prefix-stable",
                    counts=counts, identity=runtime.research_identity(),
                    cleanup=dict(interventions_cleared=True, cache_cleared=True))


def generate_one(runtime, messages, pushes=None, max_new=32, cancel=None, deadline=None,
                 *, interventions=None, patches=None):
    _deadline(deadline)
    if type(max_new) is not int or not 1 <= max_new <= 1024:
        raise ValueError("max_new must be 1..1024")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError("messages must contain 1..128 messages")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("Messages require role and content")
        _text(message["content"], "message content")
    started = time.monotonic()
    with _lease(runtime, cancel, deadline) as stop:
        if stop:
            return _not_started(runtime, stop, "generate")
        runtime.ensure_loaded()
        prompt = runtime.messages_ids(messages)
        if not prompt or len(prompt) + max_new > runtime._research_context_limit():
            raise ValueError("Prompt plus generation exceeds the bounded context")
        counts = _counts(len(prompt), max_new)
        output, ids, stop_token, reason = [], list(prompt), None, "length"
        try:
            plan = Plan(runtime, prompt, pushes, interventions, patches)
            for step in range(max_new):
                stop = interrupted(cancel, deadline)
                if stop:
                    reason = stop
                    break
                logits, _ = _forward(runtime, ids, plan, step == 0, counts)
                stop = interrupted(cancel, deadline)
                if stop:
                    reason = stop
                    break
                values = np.asarray(logits)
                if values.ndim != 1 or not np.isfinite(values).all():
                    raise ValueError("Invalid generation logits")
                token = int(values.argmax())
                if runtime._research_is_eog(token):
                    stop_token, reason = token, "eos"
                    break
                output.append(token)
                ids.append(token)
                counts["output_tokens"] += 1
        except Exception as exc:
            _failure(exc, counts, started)
            raise
        finally:
            _cleanup(runtime)
        counts["wall_seconds"] = time.monotonic() - started
        return dict(prompt_token_ids=list(prompt), output_token_ids=output,
                    text=runtime.tok.decode(output, skip_special_tokens=True),
                    stop_token_id=stop_token, finish_reason=reason,
                    completed=reason in {"eos", "length"}, eos_policy="stop-before-eog",
                    counts=counts, identity=runtime.research_identity(),
                    cleanup=dict(interventions_cleared=True, cache_cleared=True))


def capture_selected(runtime, text, selection, chat=True, prefix="", pushes=None,
                     cancel=None, deadline=None):
    _deadline(deadline)
    _text(text, "text")
    _text(prefix, "prefix")
    started = time.monotonic()
    with _lease(runtime, cancel, deadline) as stop:
        if stop:
            return _not_started(runtime, stop, "capture")
        runtime.ensure_loaded()
        rendered, specials = runtime._research_render(text, chat)
        prompt = list(runtime.tok(rendered + prefix, add_special_tokens=specials).input_ids)
        if not 1 <= len(prompt) <= runtime._research_context_limit():
            raise ValueError("Capture input exceeds context")
        rows = selection_rows(runtime, selection, len(prompt))
        counts = _counts(len(prompt))
        values, reason = [], interrupted(cancel, deadline)
        try:
            plan = Plan(runtime, prompt, pushes)
            if not reason:
                _, captured = _forward(runtime, prompt, plan, True, counts, rows)
                reason = interrupted(cancel, deadline)
                if not reason:
                    values = [float(v) for v in captured]
                    if len(values) != len(rows) or not all(math.isfinite(v) for v in values):
                        raise ValueError("Missing or nonfinite selected activation")
        except Exception as exc:
            _failure(exc, counts, started)
            raise
        finally:
            _cleanup(runtime)
        counts["wall_seconds"] = time.monotonic() - started
        result = dict(kind="mlp-selected-v1", prompt_token_ids=prompt,
                      selection=[list(r) for r in rows], values=values,
                      pushes=[list(p) for p in pushes or []], identity=runtime.research_identity(),
                      completed=reason is None, finish_reason=reason or "completed", counts=counts)
        result["sha256"] = digest(result)
        return result
