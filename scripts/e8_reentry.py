"""E8 — under P_language ablation, where and when does the fact come back?

Spec E3's note: the model can re-derive the property later in the sequence, and
re-entry would itself be evidence of a repair mechanism. Measures the Spanish
coordinate over the full (layer x position) grid with P_language ablated at
stimulus positions, against the unablated (B) run.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli
from scripts.e3_causal import parse_node, src_values, _null


@torch.no_grad()
def grid(b, ids, layers, target_id, n_pos, patches=None):
    """J-lens coordinate for the target at every (layer, position)."""
    ctx = hooks.patched(b, patches) if patches else _null()
    with ctx:
        with hooks.NodeCache(b, layers=layers) as c:
            b.model.forward(ids)
            out = np.empty((len(layers), n_pos), dtype=np.float32)
            for i, l in enumerate(layers):
                jh = lens_ops.jhat(b, target_id, l)
                out[i] = (c.resid[l][0, :n_pos].float() @ jh).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--prop", default="language")
    ap.add_argument("--recover", type=float, default=0.70)
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    prop = args.prop
    l_star = jmodel.L_STAR[prop]
    P = [parse_node(n) for n in json.load(open(f"results/e3_causal_{prop}.json"))["P_prop"]]
    pairs, _ = stimuli.load(b, prop)
    tgt = stimuli.target_ids(b, prop)["primary_id"]
    layers = list(b.lens.source_layers)
    n_pos = min(p.n_tokens for p in pairs)
    print(f"{prop}: ablating P={['/'.join(str(x) for x in n) for n in P]} at stimulus "
          f"positions; grid = {len(layers)} layers x {n_pos} positions", flush=True)

    clean, abl, ref_a = [], [], []
    for i, p in enumerate(pairs):
        with torch.no_grad():
            ca = hooks.NodeCache(b, layers=list(range(l_star + 1)))
            with ca:
                b.model.forward(p.ids_a)
        patches = {n: (p.stimulus_positions,
                       src_values(b, ca, n, p.stimulus_positions)) for n in P}
        clean.append(grid(b, p.ids_b, layers, tgt, n_pos))
        abl.append(grid(b, p.ids_b, layers, tgt, n_pos, patches))
        ref_a.append(grid(b, p.ids_a, layers, tgt, n_pos))
        del ca
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    C, A, R = np.mean(clean, 0), np.mean(abl, 0), np.mean(ref_a, 0)
    np.savez(f"results/e8_reentry_{prop}.npz", clean=C, ablated=A, condA=R,
             layers=np.array(layers), n_pos=n_pos,
             p_star=int(pairs[0].p_star), stim_start=int(pairs[0].stimulus_start))

    li = layers.index(l_star)
    p_star = pairs[0].p_star
    # recovery over depth at p*: where does the ablated run catch up with clean?
    denom = C[:, p_star] - R[:, p_star]
    frac_depth = np.where(np.abs(denom) > 1e-6, (A[:, p_star] - R[:, p_star]) /
                          np.where(np.abs(denom) > 1e-6, denom, 1), np.nan)
    later = [l for l in layers if l > l_star]
    rec_layer = next((l for l in later
                      if frac_depth[layers.index(l)] >= args.recover), None)
    # recovery over position at l*
    dpos = C[li] - R[li]
    frac_pos = np.where(np.abs(dpos) > 1e-6, (A[li] - R[li]) /
                        np.where(np.abs(dpos) > 1e-6, dpos, 1), np.nan)
    rec_pos = next((q for q in range(p_star + 1, n_pos)
                    if frac_pos[q] >= args.recover), None)

    print(f"\n  at p*={p_star}, fraction of the (B)-(A) coordinate retained under ablation:")
    for l in range(l_star, layers[-1] + 1, 4):
        print(f"    L{l:<3} {frac_depth[layers.index(l)]:>7.0%}")
    print(f"\n  re-entry by depth at p*: first layer > l* recovering >= {args.recover:.0%}"
          f" -> {rec_layer}")
    print(f"  re-entry by position at l*={l_star}: first position > p* recovering"
          f" -> {rec_pos}")
    print(f"  coordinate at the final position, last lens layer:"
          f" clean {C[-1, -1]:.3f} / ablated {A[-1, -1]:.3f} / (A) {R[-1, -1]:.3f}")
    json.dump({"property": prop, "l_star": l_star, "p_star": int(p_star),
               "P": ["/".join(str(x) for x in n) for n in P],
               "reentry_layer_at_pstar": rec_layer,
               "reentry_position_at_lstar": rec_pos,
               "retained_by_layer_at_pstar": {str(l): float(frac_depth[layers.index(l)])
                                              for l in layers},
               "retained_by_position_at_lstar": {str(q): float(frac_pos[q])
                                                 for q in range(n_pos)}},
              open(f"results/e8_reentry_{prop}.json", "w"), indent=1)
    print(f"wrote results/e8_reentry_{prop}.json")


if __name__ == "__main__":
    main()
