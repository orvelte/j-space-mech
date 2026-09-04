"""Node capture and counterfactual patching.

Node types, per spec §3.1's architecture note:
  ``("attn", layer, head)`` — a full-attention head, only at ``i % 4 == 3``;
  ``("gdn", layer)``        — a Gated DeltaNet block (recurrent, no attn pattern);
  ``("mlp", layer)``        — an MLP block.

Attention heads are captured at the *input* to ``o_proj``, after the
``sigmoid`` output gate. Since ``o_proj`` is linear and the gate is elementwise
over head dims, the per-head slice ``[h*head_dim:(h+1)*head_dim]`` maps to that
head's additive contribution to the residual stream by a fixed linear map — so a
gradient or a difference taken on the slice is equivalent to one taken on the
contribution itself, and is cheaper.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def _mixer(bundle, layer):
    block = bundle.block(layer)
    return block.self_attn if bundle.is_full_attn(layer) else block.linear_attn


class NodeCache:
    """Captures residual streams and per-node activations on one forward pass.

    Args:
        bundle: the loaded :class:`~jspace.model.Bundle`.
        layers: layers to capture at (default: all).
        build_graph: if True, make the embedding output require grad so the whole
            stack is differentiable and captured tensors are non-leaf nodes of a
            retained graph — needed for E2 attribution. If False, everything is
            detached.
        attn_patterns: also capture attention probabilities (needs
            ``attn_implementation="eager"``); used for H1 / E5.
    """

    def __init__(self, bundle, layers=None, *, build_graph=False, attn_patterns=False):
        self.bundle = bundle
        self.layers = sorted(range(bundle.n_layers) if layers is None else layers)
        self.build_graph = build_graph
        self.attn_patterns = attn_patterns
        self.resid: dict[int, torch.Tensor] = {}
        self.attn_in: dict[int, torch.Tensor] = {}
        self.gdn: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}
        self.patterns: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _keep(self, store, key, tensor):
        store[key] = tensor if self.build_graph else tensor.detach()

    def __enter__(self):
        b = self.bundle
        if self.build_graph:
            embed = b.model._embed_tokens

            def embed_hook(module, inputs, output):
                output.requires_grad_(True)
                return output

            self._handles.append(embed.register_forward_hook(embed_hook))

        for layer in self.layers:
            block = b.block(layer)

            def resid_hook(module, inputs, output, layer=layer):
                t = output if torch.is_tensor(output) else output[0]
                self._keep(self.resid, layer, t)

            self._handles.append(block.register_forward_hook(resid_hook))

            def mlp_hook(module, inputs, output, layer=layer):
                self._keep(self.mlp, layer, output)

            self._handles.append(block.mlp.register_forward_hook(mlp_hook))

            if b.is_full_attn(layer):
                def oproj_pre(module, args, layer=layer):
                    self._keep(self.attn_in, layer, args[0])

                self._handles.append(
                    block.self_attn.o_proj.register_forward_pre_hook(oproj_pre)
                )
                if self.attn_patterns:
                    def pat_hook(module, inputs, output, layer=layer):
                        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                            self.patterns[layer] = output[1].detach()

                    self._handles.append(
                        block.self_attn.register_forward_hook(pat_hook)
                    )
            else:
                def gdn_hook(module, inputs, output, layer=layer):
                    t = output if torch.is_tensor(output) else output[0]
                    self._keep(self.gdn, layer, t)

                self._handles.append(block.linear_attn.register_forward_hook(gdn_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def node_tensor(self, node) -> torch.Tensor:
        """The captured tensor a node's activation lives in, ``[B, T, ...]``."""
        kind, layer = node[0], node[1]
        if kind == "attn":
            return self.attn_in[layer]
        if kind == "gdn":
            return self.gdn[layer]
        if kind == "mlp":
            return self.mlp[layer]
        raise KeyError(kind)

    def node_slice(self, node) -> torch.Tensor:
        """A node's activation, ``[B, T, width]`` — the head slice for ``attn``."""
        t = self.node_tensor(node)
        if node[0] == "attn":
            hd = self.bundle.head_dim
            h = node[2]
            return t[..., h * hd : (h + 1) * hd]
        return t


def nodes_for(bundle, max_layer: int, *, kinds=("attn", "gdn", "mlp")) -> list[tuple]:
    """Every node at layers ``<= max_layer``, in layer order."""
    out = []
    for layer in range(max_layer + 1):
        if "attn" in kinds and bundle.is_full_attn(layer):
            out += [("attn", layer, h) for h in range(bundle.n_heads)]
        if "gdn" in kinds and not bundle.is_full_attn(layer):
            out.append(("gdn", layer))
        if "mlp" in kinds:
            out.append(("mlp", layer))
    return out


@contextmanager
def patched(bundle, patches):
    """Counterfactual ablation: overwrite node activations at given positions.

    Args:
        patches: ``{node: (positions, values)}``. ``values`` is broadcastable to
            ``[len(positions), width]`` and is written into the node's activation
            at those sequence positions. Values come from the corrupted-condition
            run — never zeros (spec §4 E3, bad-null checklist).
    """
    handles = []
    by_layer: dict[tuple, list] = {}
    for node, (positions, values) in patches.items():
        by_layer.setdefault((node[0], node[1]), []).append((node, positions, values))

    try:
        for (kind, layer), entries in by_layer.items():
            block = bundle.block(layer)

            def apply(t, entries=entries):
                # During cached generation the prompt pass sees every position but
                # each subsequent pass sees only the new token, so the patch indices
                # do not exist. Leave those passes untouched: the intervention is
                # defined on prompt positions, and the recurrent/KV state carrying
                # it forward is already established by the prompt pass.
                if t.shape[1] <= max(max(pos) for _, pos, _ in entries):
                    return t
                t = t.clone()
                for node, positions, values in entries:
                    idx = torch.as_tensor(positions, device=t.device, dtype=torch.long)
                    v = values.to(t.dtype).to(t.device)
                    if node[0] == "attn":
                        hd = bundle.head_dim
                        h = node[2]
                        t[:, idx, h * hd : (h + 1) * hd] = v
                    else:
                        t[:, idx, :] = v
                return t

            if kind == "attn":
                def pre(module, args, apply=apply):
                    return (apply(args[0]),) + tuple(args[1:])

                handles.append(block.self_attn.o_proj.register_forward_pre_hook(pre))
            elif kind == "mlp":
                def fwd(module, inputs, output, apply=apply):
                    return apply(output)

                handles.append(block.mlp.register_forward_hook(fwd))
            elif kind == "gdn":
                def fwd_gdn(module, inputs, output, apply=apply):
                    if torch.is_tensor(output):
                        return apply(output)
                    return (apply(output[0]),) + tuple(output[1:])

                handles.append(block.linear_attn.register_forward_hook(fwd_gdn))
            else:
                raise KeyError(kind)
        yield
    finally:
        for h in handles:
            h.remove()
