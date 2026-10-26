"""ctypes boundary to the version-pinned native bridge; no saved weight writes."""

import ctypes as C
import fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from gguf_metadata import describe, metadata, identity

ROOT = Path(__file__).resolve().parents[1]
F32 = C.POINTER(C.c_float)
I32 = C.POINTER(C.c_int)
D64 = C.POINTER(C.c_double)


class Push(C.Structure):
    _fields_ = [
        ("layer", C.c_int),
        ("neuron", C.c_int),
        ("mode", C.c_int),
        ("value", C.c_float),
    ]


def ptr(a, dtype=F32):
    return None if a is None else a.ctypes.data_as(dtype)


class Bridge:
    def __init__(self, path, hooks=True, *, weights_only=False):
        lib = ROOT / ".synapse/libsynapse-llama.so"
        if not lib.exists():
            raise RuntimeError(
                "Build the GGUF bridge first: python scripts/build_gguf_bridge.py"
            )
        self.lib = C.CDLL(str(lib))
        signatures = {
            "error": (C.c_char_p, []),
            "open": (C.c_void_p, [C.c_char_p, C.c_char_p, C.c_int, C.c_bool]),
            "open_inspection": (C.c_void_p, [C.c_char_p, C.c_char_p]),
            "has_context": (C.c_bool, [C.c_void_p]),
            "close": (None, [C.c_void_p]),
            "vocab": (C.c_int, [C.c_void_p]),
            "tokenize": (
                C.c_int,
                [C.c_void_p, C.c_char_p, C.c_int, I32, C.c_int, C.c_bool],
            ),
            "piece": (C.c_int, [C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_bool]),
            "eog": (C.c_bool, [C.c_void_p, C.c_int]),
            "eval": (
                C.c_int,
                [
                    C.c_void_p,
                    I32,
                    C.c_int,
                    C.c_bool,
                    C.c_bool,
                    C.c_bool,
                    C.c_int,
                    C.c_int,
                    C.POINTER(Push),
                    C.c_int,
                    F32,
                    F32,
                    F32,
                ],
            ),
            "column": (C.c_int, [C.c_void_p, C.c_int, C.c_int, F32]),
            "lens": (C.c_int, [C.c_void_p, C.c_int, C.c_int, F32]),
            "rows": (C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int, F32]),
            "connections": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int, F32, F32]),
        }
        for name, (ret, args) in signatures.items():
            f = getattr(self.lib, "syn_" + name)
            f.restype, f.argtypes = ret, args
        if hasattr(self.lib, "syn_research"):
            self.lib.syn_research.restype = C.c_int
            self.lib.syn_research.argtypes = [
                C.c_void_p, I32, C.c_int, C.c_bool, C.c_int, C.c_int,
                F32, C.c_float, F32, D64, F32,
            ]
        if hasattr(self.lib, "syn_research_factors"):
            self.lib.syn_research_factors.restype = C.c_int
            self.lib.syn_research_factors.argtypes = self.lib.syn_research.argtypes + [F32, C.c_int]
        if hasattr(self.lib, "syn_shape"):
            self.lib.syn_shape.restype = C.c_int
            self.lib.syn_shape.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int64)]
        self.model_path = Path(path).expanduser().resolve()
        self.identity, self.model_key = identity(self.model_path)
        self.arch = describe(path)
        meta = metadata(path)
        self.chat_template = meta.get("tokenizer.chat_template")
        self.rms_epsilon = meta.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        self.weights_only = weights_only
        self.lease = None
        if not weights_only:
            self.lease = open(ROOT / ".synapse/local-model-cpu.lock", "a")
            try:
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.lease.close()
                raise RuntimeError(
                    "Another local Qwen runtime owns the CPU model slot. Stop qwen-lab or Synapse before switching."
                ) from None
        args = (str(path).encode(), str(ROOT / "runtime/llama/llama-b11669").encode())
        self.handle = (self.lib.syn_open_inspection(*args) if weights_only
                       else self.lib.syn_open(*args, 4, hooks))
        if not self.handle:
            if self.lease:
                self.lease.close()
            raise RuntimeError(self.lib.syn_error().decode())
        self.vocab = self.lib.syn_vocab(self.handle)
        self._pieces = {}

    def close(self):
        if self.handle:
            self.lib.syn_close(self.handle)
            self.handle = None
            if self.lease:
                self.lease.close()

    def check(self, rc):
        if rc < 0:
            raise RuntimeError(self.lib.syn_error().decode())

    def tokenize(self, text, special=True):
        b = text.encode()
        ids = np.empty(len(b) + 16, dtype=np.int32)
        n = self.lib.syn_tokenize(
            self.handle, b, len(b), ptr(ids, I32), len(ids), special
        )
        if n < 0:
            ids = np.empty(-n, dtype=np.int32)
            n = self.lib.syn_tokenize(
                self.handle, b, len(b), ptr(ids, I32), len(ids), special
            )
        if n < 0:
            raise ValueError("Tokenizer failed")
        return ids[:n].tolist()

    def piece(self, token, special=True):
        key = (token, special)
        if key not in self._pieces:
            buf = C.create_string_buffer(256)
            n = self.lib.syn_piece(self.handle, token, buf, len(buf), special)
            if n < 0:
                buf = C.create_string_buffer(-n)
                n = self.lib.syn_piece(self.handle, token, buf, len(buf), special)
            if n < 0:
                raise ValueError("Token decoding failed")
            self._pieces[key] = buf.raw[:n]
        return self._pieces[key]

    def run(
        self,
        ids,
        reset=True,
        read=None,
        full=False,
        residual=False,
        pushes=(),
        observe=False,
    ):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        L, N = self.arch["n_layers"], self.arch["inter"]
        acts = (
            np.zeros((L, len(ids), N), np.float32)
            if full
            else np.zeros(len(ids), np.float32)
            if read
            else None
        )
        resid = np.zeros((L, len(ids)), np.float32) if residual else None
        logits = np.empty(self.vocab, np.float32)
        ps = (Push * len(pushes))(
            *(
                Push(l, -1 if n is None else n, int(mode == "mul"), v)
                for l, n, mode, v in pushes
            )
        )
        l, n = read or (-1, -1)
        self.check(
            self.lib.syn_eval(
                self.handle,
                ptr(ids, I32),
                len(ids),
                reset,
                observe,
                full,
                l,
                n,
                ps,
                len(ps),
                ptr(acts),
                ptr(resid),
                ptr(logits),
            )
        )
        return logits, acts, resid

    def column(self, l, n):
        out = np.empty(self.arch["hidden"], np.float32)
        self.check(self.lib.syn_column(self.handle, l, n, ptr(out)))
        return out

    def research(self, ids, *, reset=True, layer=None, position=-1,
                 direction=None, strength=0.0, factors=None):
        """Selected residual capture and reversible projection of residual writes.

        No neuron pushes are combined with this operation. The shared readout is
        untouched. The native ABI verifies every intended writer on every token.
        """
        from residual_projection import unit_direction, bounded_strength
        if not hasattr(self.lib, "syn_research"):
            raise RuntimeError("Residual research needs a rebuilt bridge and server restart")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        if ids.ndim != 1 or not 1 <= len(ids) <= 2048:
            raise ValueError("Residual research needs 1..2048 tokens")
        hidden, layers = self.arch["hidden"], self.arch["n_layers"]
        if layer is not None and (type(layer) is not int or not 0 <= layer < layers):
            raise ValueError("Invalid residual capture layer")
        if type(position) is not int or not -len(ids) <= position < len(ids):
            raise ValueError("Invalid residual token position")
        position %= len(ids)
        strength = bounded_strength(strength)
        u = unit_direction(direction, hidden) if direction is not None else None
        if u is None and strength != 0:
            raise ValueError("Nonzero projection requires a direction")
        vector = np.empty(hidden, np.float32) if layer is not None else None
        stats = np.zeros((1 + 2 * layers, 3), np.float64)
        logits = np.empty(self.vocab, np.float32)
        args = (
            self.handle, ptr(ids, I32), len(ids), reset,
            -1 if layer is None else layer, position, ptr(u), strength,
            ptr(vector), ptr(stats, D64), ptr(logits),
        )
        if factors is None:
            self.check(self.lib.syn_research(*args))
        else:
            if not hasattr(self.lib, "syn_research_factors"):
                raise RuntimeError("Weight factors require a rebuilt bridge and restart")
            factors = np.ascontiguousarray(factors, dtype=np.float32)
            if factors.ndim != 1 or factors.size > 2_000_000 or not np.isfinite(factors).all():
                raise ValueError("Invalid bounded weight factors")
            self.check(self.lib.syn_research_factors(*args, ptr(factors), factors.size))
        return logits, vector, stats

    def lens(self, l, n):
        out = np.empty(self.vocab, np.float32)
        self.check(self.lib.syn_lens(self.handle, l, n, ptr(out)))
        return out

    def weight_rows(self, name, start, count, width=None):
        width = self.arch["hidden"] if width is None else width
        if not 1 <= count <= 2048 or not 1 <= width <= max(
            self.arch["hidden"], self.arch["inter"]
        ):
            raise ValueError("Invalid weight-row allocation")
        out = np.empty((count, width), np.float32)
        self.check(
            self.lib.syn_rows(self.handle, name.encode(), start, count, width, ptr(out))
        )
        return out

    def weight_shape(self, name):
        if not hasattr(self.lib, "syn_shape"):
            raise RuntimeError("Tensor shapes need a rebuilt bridge and server restart")
        shape = (C.c_int64 * 4)()
        self.check(self.lib.syn_shape(self.handle, name.encode(), shape))
        return tuple(shape)

    def connection_vectors(self, layer, n):
        shape = (self.arch["n_layers"], self.arch["inter"])
        gate, up = np.empty(shape, np.float32), np.empty(shape, np.float32)
        self.check(
            self.lib.syn_connections(
                self.handle, layer, n, gate.size, ptr(gate), ptr(up)
            )
        )
        return gate, up


class Tokenizer:
    def __init__(self, bridge):
        self.bridge = bridge
        self.chat_template = bridge.chat_template

    def __len__(self):
        return self.bridge.vocab

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return SimpleNamespace(input_ids=self.bridge.tokenize(text, add_special_tokens))

    def decode(self, ids, skip_special_tokens=False):
        return b"".join(
            self.bridge.piece(int(t), not skip_special_tokens) for t in ids
        ).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, **kwargs):
        from transformers.utils.chat_template_utils import render_jinja_template

        rendered, _ = render_jinja_template(
            [messages],
            chat_template=self.chat_template,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.bridge.tokenize(rendered[0], False)
