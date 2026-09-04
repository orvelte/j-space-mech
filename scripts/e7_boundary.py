"""E7 — which layers carry "report this" across the question/stimulus boundary?

The instruction can only reach stimulus positions by crossing from question
positions, and in this hybrid the crossing happens either inside a full-attention
layer (via K/V at question positions) or inside a GDN recurrent state (via the
state accumulated over the question span). This localises that crossing.

**One uniform intervention for both layer types.** At layer l:

  1. replace the token mixer's *input* rows at question positions with the (A)
     run's, so everything the mixer accumulates from the instruction -- the GDN
     recurrent state at the boundary, or the attention K/V the stimulus reads --
     comes from (A);
  2. restore the mixer's *output* rows at question positions to their (B) values,
     so the residual stream at question positions is untouched downstream.

Step 2 is what makes this a single-layer claim: without it, the intervention also
perturbs the question-position residual that later layers read, and the effect
could not be attributed to layer l. What remains is exactly the information layer
l's mixer carries from the question into stimulus positions.

For a GDN layer this is equivalent to patching the boundary recurrent state,
since that state is a deterministic function of the inputs over the question
span, and the stimulus inputs are unchanged.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli


def mixer(b, l):
    blk = b.block(l)
    return blk.self_attn if b.is_full_attn(l) else blk.linear_attn


class MixerIO:
    """Records each layer's token-mixer input and output."""

    def __init__(self, b, layers):
        self.b, self.layers = b, layers
        self.inp, self.out = {}, {}
        self._h = []

    def __enter__(self):
        for l in self.layers:
            m = mixer(self.b, l)

            def pre(mod, a, kw, l=l):
                h = kw.get("hidden_states", a[0] if a else None)
                self.inp[l] = h.detach()

            def post(mod, a, kw, o, l=l):
                t = o if torch.is_tensor(o) else o[0]
                self.out[l] = t.detach()

            self._h.append(m.register_forward_pre_hook(pre, with_kwargs=True))
            self._h.append(m.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *e):
        for h in self._h:
            h.remove()
        self._h = []


class BoundaryPatch:
    """Patch layer l's mixer input at question positions; hold its output there."""

    def __init__(self, b, l, q_pos, src_in, keep_out):
        self.b, self.l, self.q = b, l, q_pos
        self.src_in, self.keep_out = src_in, keep_out
        self._h = []

    def __enter__(self):
        m = mixer(self.b, self.l)
        idx = torch.as_tensor(self.q, device="cuda", dtype=torch.long)

        def pre(mod, a, kw):
            h = kw.get("hidden_states", a[0] if a else None)
            h = h.clone()
            h[:, idx] = self.src_in.to(h.dtype)
            if "hidden_states" in kw:
                kw["hidden_states"] = h
                return a, kw
            return (h,) + tuple(a[1:]), kw

        def post(mod, a, kw, o):
            t = o if torch.is_tensor(o) else o[0]
            t = t.clone()
            t[:, idx] = self.keep_out.to(t.dtype)
            return t if torch.is_tensor(o) else (t,) + tuple(o[1:])

        self._h.append(m.register_forward_pre_hook(pre, with_kwargs=True))
        self._h.append(m.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *e):
        for h in self._h:
            h.remove()
        self._h = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--prop", default="language")
    ap.add_argument("--max-layer", type=int, default=None)
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    prop = args.prop
    l_star = jmodel.L_STAR[prop]
    layers = list(range(min(args.max_layer or l_star, l_star) + 1))
    pairs, _ = stimuli.load(b, prop)
    tgt = stimuli.target_ids(b, prop)["primary_id"]
    print(f"{prop}: l*={l_star}, patching the boundary at each of layers "
          f"{layers[0]}–{layers[-1]}, {len(pairs)} stimuli", flush=True)

    @torch.no_grad()
    def coord(ids, p_star):
        with hooks.NodeCache(b, layers=[l_star]) as c:
            b.model.forward(ids)
            return float(lens_ops.coordinate(b, c.resid[l_star][0, p_star], l_star, tgt))

    per = []
    for p in pairs:
        with torch.no_grad():
            with MixerIO(b, layers) as ia:
                b.model.forward(p.ids_a)
                a_in = {l: ia.inp[l][:, :p.stimulus_start].clone() for l in layers}
            with MixerIO(b, layers) as ib:
                b.model.forward(p.ids_b)
                b_out = {l: ib.out[l][:, :p.stimulus_start].clone() for l in layers}
        per.append({"pair": p, "a_in": a_in, "b_out": b_out,
                    "mb": coord(p.ids_b, p.p_star), "ma": coord(p.ids_a, p.p_star)})
    mb, ma = np.mean([d["mb"] for d in per]), np.mean([d["ma"] for d in per])
    gap = float(mb - ma)
    print(f"  (B)-(A) gap = {gap:.4f}", flush=True)

    rows = []
    for l in layers:
        ms = []
        for d in per:
            p = d["pair"]
            with torch.no_grad():
                with BoundaryPatch(b, l, list(range(p.stimulus_start)),
                                   d["a_in"][l], d["b_out"][l]):
                    ms.append(coord(p.ids_b, p.p_star))
        m = float(np.mean(ms))
        se = float(np.std(ms, ddof=1) / np.sqrt(len(ms)))
        frac = (mb - m) / gap if gap else float("nan")
        rows.append({"layer": l, "kind": "attn" if b.is_full_attn(l) else "gdn",
                     "mean_coord": m, "se": se, "frac_of_gap_killed": float(frac)})
        print(f"  L{l:<3} {rows[-1]['kind']:<5} coord {m:.4f}  kills {frac:>7.1%} of the gap",
              flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r["frac_of_gap_killed"])
    print("\n  top 8 boundary-carrying layers:")
    for r in rows_sorted[:8]:
        print(f"    L{r['layer']:<3} {r['kind']:<5} {r['frac_of_gap_killed']:>7.1%}")
    attn_layers = [r for r in rows if r["kind"] == "attn"]
    gdn_layers = [r for r in rows if r["kind"] == "gdn"]
    summary = {
        "property": prop, "l_star": l_star, "BA_gap": gap, "per_layer": rows,
        "top_layers": rows_sorted[:8],
        "max_attn": max(r["frac_of_gap_killed"] for r in attn_layers),
        "max_gdn": max(r["frac_of_gap_killed"] for r in gdn_layers),
        "sum_attn": float(sum(r["frac_of_gap_killed"] for r in attn_layers)),
        "sum_gdn": float(sum(r["frac_of_gap_killed"] for r in gdn_layers)),
    }
    print(f"\n  best single attention layer: {summary['max_attn']:.1%}")
    print(f"  best single GDN layer:       {summary['max_gdn']:.1%}")
    json.dump(summary, open(f"results/e7_boundary_{prop}.json", "w"), indent=1)
    print(f"wrote results/e7_boundary_{prop}.json")


if __name__ == "__main__":
    main()
