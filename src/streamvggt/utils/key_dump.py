"""Per-step dump of the current frame's keys in every global attention layer (EpiCache-style layer sensitivity).

Forward hooks on ``aggregator.global_blocks[i].attn.k_norm`` record the keys the current frame produces in each
global layer (after q/k normalisation, before RoPE; RoPE rotates both runs identically, so cosines are unaffected).
A key is computed once, when its frame enters, from that layer's input, so a run with eviction and a full-cache run of
the same sequence differ only through the attention of earlier layers over a reduced cache.

Usage from a launcher: ``install(model)`` once, then per scene ``set_scene(path)`` before inference and ``flush()``
after it. The file holds ``keys[layer]`` = float16 tensor [steps, heads, tokens, head_dim].
"""

import os
from typing import Optional

import torch

_STATE = {"path": None, "keys": None, "installed": False}


def install(model) -> None:
    if _STATE["installed"]:
        return
    blocks = model.aggregator.global_blocks
    for layer, block in enumerate(blocks):
        def hook(_module, _inputs, output, layer=layer):
            if _STATE["keys"] is not None:
                _STATE["keys"][layer].append(output.detach()[0].to("cpu", torch.float16))
        block.attn.k_norm.register_forward_hook(hook)
    _STATE["installed"] = True
    _STATE["num_layers"] = len(blocks)


def set_scene(path: Optional[str]) -> None:
    _STATE["path"] = path
    _STATE["keys"] = [[] for _ in range(_STATE.get("num_layers", 0))] if path else None


def flush() -> None:
    if _STATE["path"] is None or _STATE["keys"] is None:
        return
    os.makedirs(os.path.dirname(_STATE["path"]), exist_ok=True)
    keys = [torch.stack(per_layer) if per_layer else torch.empty(0) for per_layer in _STATE["keys"]]
    torch.save({"keys": keys}, _STATE["path"])
    _STATE["keys"] = None
    _STATE["path"] = None
