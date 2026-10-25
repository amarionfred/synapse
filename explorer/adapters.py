"""Structural adapters for dense Hugging Face decoders; no family is silently assumed.

The intervention coordinate is the *input* to the MLP output projection, whatever
activation/gating preceded it. Packed quantization, MoE and nonuniform widths fail closed.
"""
from dataclasses import dataclass
import inspect
from types import SimpleNamespace

import torch
from torch import nn


class UnsupportedArchitecture(ValueError):
    pass


def at(obj, path):
    for name in path.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def first(obj, paths):
    for p in paths:
        value = at(obj, p)
        if value is not None:
            return p, value
    return None, None


@dataclass
class Projection:
    module: nn.Module
    path: str
    transposed: bool = False

    @classmethod
    def wrap(cls, module, path):
        # Conv1D in GPT-2 stores [in, out]; Linear stores [out, in].
        conv = type(module).__name__ == "Conv1D" and type(module).__module__.startswith("transformers.")
        if not (type(module) is nn.Linear or conv) or module.weight.ndim != 2:
            raise UnsupportedArchitecture(f"{path}: {type(module).__name__} is not a supported floating projection")
        if not module.weight.is_floating_point():
            raise UnsupportedArchitecture(f"{path}: packed weights require a dequantization adapter")
        return cls(module, path, conv)

    @property
    def weight(self):
        return self.module.weight.T if self.transposed else self.module.weight

    @property
    def width(self):
        return self.weight.shape[1]


class DenseAdapter:
    def __init__(self, model):
        self.model = model
        path, layers = first(model, ["model.language_model.layers", "model.layers", "language_model.model.layers",
                                     "transformer.h", "gpt_neox.layers", "model.decoder.layers"])
        if layers is None or not len(layers):
            raise UnsupportedArchitecture("No supported dense decoder stack found. Add an adapter for this architecture.")
        self.path, self.layers = path, layers
        self.base = at(model, path.rsplit(".", 1)[0])
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        if any(getattr(cfg, n, 1) > 1 for n in ("num_local_experts", "num_experts", "n_routed_experts")
               if getattr(cfg, n, None) is not None):
            raise UnsupportedArchitecture("MoE needs expert-specific coordinates")
        self.down, self.attention = [], []
        for i, layer in enumerate(layers):
            p, mod = first(layer, ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "fc2", "mlp.fc2"])
            if mod is None:
                raise UnsupportedArchitecture(f"Layer {i} has no supported dense MLP output projection")
            self.down.append(Projection.wrap(mod, f"{path}.{i}.{p}"))
            hp, hm = first(layer, ["self_attn.o_proj", "attn.c_proj", "attention.dense",
                                  "self_attn.out_proj", "linear_attn.out_proj"])
            head = None
            if hm is not None:
                try:
                    proj = Projection.wrap(hm, hp)
                    owner = at(layer, hp.rsplit(".", 1)[0])
                    nh = (getattr(owner, "num_v_heads", None) if hp.startswith("linear_attn") else None)
                    hd = getattr(owner, "head_dim", None)
                    nh = nh or (proj.width // hd if hd else getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
                    if nh and proj.width % nh == 0:
                        head = (proj, nh)
                except UnsupportedArchitecture:
                    pass
            self.attention.append(head)
        shapes = {tuple(p.weight.shape) for p in self.down}
        if len(shapes) != 1:
            raise UnsupportedArchitecture("Nonuniform MLP widths need a ragged map/activation adapter")
        _, self.norm = first(self.base, ["norm", "ln_f", "final_layer_norm", "final_layernorm"])
        self.unembed = model.get_output_embeddings()
        project_out = getattr(self.base, "project_out", None)
        if project_out is not None and self.unembed is not None:
            self.unembed = nn.Sequential(project_out, self.unembed)
        self.lens_supported = self.norm is not None and self.unembed is not None
        # Direct-path score only has defined semantics here for separate gate/up projections and RMSNorm.
        self.connections_supported = all(
            isinstance(at(l, "mlp.gate_proj"), nn.Linear) and isinstance(at(l, "mlp.up_proj"), nn.Linear)
            and at(l, "post_attention_layernorm") is not None
            and at(l, "pre_feedforward_layernorm") is None
            and "RMSNorm" in type(l.post_attention_layernorm).__name__ for l in layers)
        self.forward_keys = inspect.signature(model.forward).parameters
        self.view = SimpleNamespace(layers=layers, norm=self.norm)

    def check(self, layer, neuron=None):
        if not 0 <= layer < len(self.layers):
            raise ValueError(f"Layer {layer} outside 0..{len(self.layers)-1}")
        if neuron is not None and not 0 <= neuron < self.down[layer].width:
            raise ValueError(f"Neuron {neuron} outside 0..{self.down[layer].width-1}")

    def norm_gain(self, layer):
        norm = self.layers[layer].post_attention_layernorm
        w = norm.weight.float()
        name = type(norm).__name__
        # Transformers' Gemma and Qwen3.5 RMSNorm parameters are zero-centered.
        return 1 + w if name.startswith(("Gemma", "Qwen3_5")) else w

    def logits(self, **kwargs):
        if "logits_to_keep" in self.forward_keys:
            kwargs["logits_to_keep"] = 1
        if "position_ids" not in self.forward_keys:
            kwargs.pop("position_ids", None)
        return self.model(**kwargs)

    def describe(self):
        return {"adapter": "dense-decoder", "stack": self.path, "layers": len(self.layers),
                "neurons_per_layer": self.down[0].width,
                "capabilities": {"activations": True, "additive_steering": True, "multiplicative_steering": True,
                                 "logit_lens": self.lens_supported, "weight_connections": self.connections_supported,
                                 "attention_heads": [h[1] if h else 0 for h in self.attention]},
                "dose_units": "median layer residual norm / output-projection column norm; later norms may change the residual effect",
                "projections": [p.path for p in self.down]}
