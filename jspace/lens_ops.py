"""J-lens readout: lens directions, coordinates, and vocabulary ranks.

Spec §3.1 defines the readout direction as ``Jhat_l[t] = normalise(J_l^T W_U[t])``
and the coordinate as ``c_t(l,p) = <h_l(p), Jhat_l[t]>``.

One refinement over the literal formula (see DECISIONS.md D6): the model's
unembedding is ``lm_head(final_norm(x))``, and ``final_norm`` is an RMSNorm with a
learned per-dimension gain ``g``. So the lens's own readout of token ``t`` is
``(g * W_U[t]) . (J_l h) / rms(J_l h)``. Folding ``g`` into ``W_U`` makes the
coordinate agree with ``lens.apply``'s ranking exactly (the ``1/rms`` factor is
constant across tokens and so cannot change ranks), while keeping the coordinate a
*linear* functional of ``h`` — which attribution patching in E2 requires.
"""

from __future__ import annotations

import os

import torch

_VNORM_CACHE_DIR = os.environ.get(
    "JSPACE_CACHE", "/workspace/.cache/jspace"
)


def final_norm_gain(bundle) -> torch.Tensor:
    """The learned per-dimension gain of the final pre-unembed RMSNorm."""
    norm = getattr(bundle.model, "_final_norm", None)
    if norm is None:  # fall back to the HF path
        norm = bundle.hf.model.norm
    return norm.weight.detach().float()


def w_eff(bundle) -> torch.Tensor:
    """``W_U`` with the final-norm gain folded in: ``[vocab, d_model]``, bf16 on GPU."""
    if not hasattr(bundle, "_w_eff"):
        W_U = bundle.hf.lm_head.weight.detach()
        g = final_norm_gain(bundle).to(W_U.device)
        bundle._w_eff = (W_U.float() * g).to(torch.bfloat16)
    return bundle._w_eff


def jhat(bundle, token_id: int, layer: int, *, normalise: bool = True) -> torch.Tensor:
    """Lens direction for ``token_id`` at ``layer``: ``normalise(J_l^T (g * W_U[t]))``.

    Returns a fp32 ``[d_model]`` vector in layer-``layer`` residual space. The
    coordinate is its inner product with the residual.
    """
    J = bundle.lens.jacobians[layer].cuda()  # [d_model, d_model], fp32
    w = w_eff(bundle)[token_id].float()  # [d_model]
    v = J.T @ w
    return v / v.norm() if normalise else v


def transport(bundle, h: torch.Tensor, layer: int) -> torch.Tensor:
    """``J_l @ h`` for residuals ``h`` of shape ``[..., d_model]`` (fp32)."""
    J = bundle.lens.jacobians[layer].cuda()
    return h.float() @ J.T


def lens_scores(bundle, h: torch.Tensor, layer: int, *, use_jacobian: bool = True):
    """Unnormalised lens scores over the whole vocabulary: ``[..., vocab]``.

    ``use_jacobian=False`` is the logit-lens control (spec §3.1, deflationary #2).
    """
    x = transport(bundle, h, layer) if use_jacobian else h.float()
    return (x.to(torch.bfloat16) @ w_eff(bundle).T).float()


def direction_norms(bundle, layer: int, *, chunk: int = 16384) -> torch.Tensor:
    """``||J_l^T (g * W_U[t])||`` for every vocab token, cached to disk.

    Needed to turn unnormalised lens scores into the spec's unit-normalised
    coordinates for *all* tokens at once, which is what the top-25 rank criterion
    is defined over.
    """
    os.makedirs(_VNORM_CACHE_DIR, exist_ok=True)
    path = os.path.join(_VNORM_CACHE_DIR, f"vnorm_{bundle.variant}_L{layer}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cuda")
    J = bundle.lens.jacobians[layer].cuda()
    W = w_eff(bundle)
    out = torch.empty(W.shape[0], device="cuda", dtype=torch.float32)
    for i in range(0, W.shape[0], chunk):
        block = W[i : i + chunk].float() @ J  # (W_eff J)[t] == (J^T W_eff[t])^T
        out[i : i + chunk] = block.norm(dim=-1)
    torch.save(out, path)
    return out


def coordinates(bundle, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Unit-normalised J-lens coordinates over the whole vocabulary: ``[..., vocab]``."""
    return lens_scores(bundle, h, layer) / direction_norms(bundle, layer)


def coordinate(bundle, h: torch.Tensor, layer: int, token_id: int) -> torch.Tensor:
    """The spec's ``c_t(l, p)`` for a single token — linear in ``h``, so this is the
    metric ``m`` that E2 backpropagates."""
    return h.float() @ jhat(bundle, token_id, layer)


def rank_of(scores: torch.Tensor, token_id: int) -> torch.Tensor:
    """0-indexed rank of ``token_id`` among all vocab tokens (0 = top-1)."""
    return (scores > scores[..., token_id : token_id + 1]).sum(dim=-1)


def logit_direction(bundle, token_id: int) -> torch.Tensor:
    """Logit-lens direction for ``token_id``: ``normalise(g * W_U[t])``.

    The deflationary control of spec §3.1 / E0.4 — same functional form as
    :func:`jhat` but without the Jacobian transport.
    """
    w = w_eff(bundle)[token_id].float()
    return w / w.norm()
