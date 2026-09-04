"""Direct-effect decomposition and target mobility (DECISIONS.md D14).

Part A -- for each block, split the ablation effect into:
  direct   : the change in the block's *own* additive contribution to h_l*(p*),
             projected onto Jhat_l*. This is the residual-stream path with no
             further processing -- i.e. the block writing the readout direction.
  indirect : the remainder of the measured Δm, routed through later layers.

Part B -- re-run block attribution and patching against a metric integrated over
the whole divergence window (sum of the coordinate at p* across those layers), so
blocks that happen to write at l* are not privileged.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli
from scripts.e3_causal import src_values, _null


def window_metric(b, cache, p_star, layers, target_id):
    """Sum of the J-lens coordinate at p* over `layers` -- linear in the residuals."""
    return sum(lens_ops.coordinate(b, cache.resid[l][0, p_star], l, target_id)
               for l in layers)


@torch.no_grad()
def measure(b, ids, p_star, layers, target_id, patches=None):
    ctx = hooks.patched(b, patches) if patches else _null()
    with ctx:
        with hooks.NodeCache(b, layers=layers) as c:
            b.model.forward(ids)
            return float(window_metric(b, c, p_star, layers, target_id))


def block_members(b, kind, l):
    return ([("attn", l, h) for h in range(b.n_heads)] if kind == "attn"
            else [(kind, l)])


def contribution(b, cache, node, pos):
    """A node's additive contribution to the residual stream at `pos`, [d_model]."""
    if node[0] == "attn":
        hd = b.head_dim
        sl = cache.attn_in[node[1]][0, pos]                     # [n_heads*hd]
        W = b.block(node[1]).self_attn.o_proj.weight            # [d_model, n_heads*hd]
        return (W.float() @ sl.float())
    store = cache.gdn if node[0] == "gdn" else cache.mlp
    return store[node[1]][0, pos].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--prop", default="language")
    ap.add_argument("--window", default="24-32")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    prop = args.prop
    l_star = jmodel.L_STAR[prop]
    lo, hi = (int(x) for x in args.window.split("-"))
    win = list(range(lo, hi + 1))
    pairs, _ = stimuli.load(b, prop)
    tgt = stimuli.target_ids(b, prop)["primary_id"]
    e5 = json.load(open("results/e5_granularity.json"))[prop]
    ranked = [r for r in e5["blocks"]][:args.top]
    print(f"{prop}: l*={l_star}, window {lo}-{hi}, {len(pairs)} stimuli", flush=True)

    # cache (A) activations and per-passage baselines for both metrics
    per = []
    for p in pairs:
        with torch.no_grad():
            ca = hooks.NodeCache(b, layers=list(range(hi + 1)))
            with ca:
                b.model.forward(p.ids_a)
            cb = hooks.NodeCache(b, layers=list(range(hi + 1)))
            with cb:
                b.model.forward(p.ids_b)
        per.append({"pair": p, "cache_a": ca, "cache_b": cb})

    jhat = lens_ops.jhat(b, tgt, l_star)

    # ---- Part A: direct vs indirect at l* ------------------------------
    print("\nPart A -- direct/indirect split at l*")
    rowsA = []
    for r in ranked:
        kind, l = r["kind"], r["layer"]
        directs = []
        for d in per:
            p = d["pair"]
            dv = torch.zeros(b.d_model, device="cuda")
            for n in block_members(b, kind, l):
                dv += (contribution(b, d["cache_a"], n, p.p_star)
                       - contribution(b, d["cache_b"], n, p.p_star))
            directs.append(float(dv @ jhat))
        direct = float(np.mean(directs))
        total = r["true_delta_m"]
        frac = direct / total if total else float("nan")
        rowsA.append({"block": r["block"], "true_delta_m": total, "direct": direct,
                      "indirect": total - direct, "direct_fraction": frac})
        print(f"  {r['block']:<10} Δm {total:>8.4f} = direct {direct:>8.4f} "
              f"+ indirect {total - direct:>8.4f}   ({frac:>6.0%} direct)", flush=True)
    med = float(np.median([r["direct_fraction"] for r in rowsA]))
    print(f"  median direct fraction of top {args.top}: {med:.0%}")

    # ---- Part B: integrated-window metric -------------------------------
    print(f"\nPart B -- block effects against the integrated metric over {lo}-{hi}")
    mb = [measure(b, d["pair"].ids_b, d["pair"].p_star, win, tgt) for d in per]
    ma = [measure(b, d["pair"].ids_a, d["pair"].p_star, win, tgt) for d in per]
    gap = float(np.mean(mb) - np.mean(ma))
    print(f"  integrated (B)-(A) gap = {gap:.4f}", flush=True)

    blocks = []
    for l in range(hi + 1):
        blocks.append(("attn", l) if b.is_full_attn(l) else ("gdn", l))
        blocks.append(("mlp", l))
    rowsB = []
    for kind, l in blocks:
        members = block_members(b, kind, l)
        ms = []
        for d in per:
            p = d["pair"]
            patches = {n: (p.stimulus_positions,
                           src_values(b, d["cache_a"], n, p.stimulus_positions))
                       for n in members}
            ms.append(measure(b, p.ids_b, p.p_star, win, tgt, patches))
        delta = float(np.mean(ms)) - float(np.mean(mb))
        rowsB.append({"block": f"{kind}/{l}", "kind": kind, "layer": l,
                      "true_delta_m": delta, "frac_of_gap": delta / gap if gap else None})
    rowsB.sort(key=lambda r: r["true_delta_m"])
    print("  top 10 under the integrated metric:")
    for r in rowsB[:10]:
        print(f"    {r['block']:<10} Δm {r['true_delta_m']:>8.4f}  {r['frac_of_gap']:>7.0%}")

    A = {r["block"] for r in ranked}
    B = {r["block"] for r in rowsB[:args.top]}
    jac = len(A & B) / len(A | B)
    print(f"\n  top-{args.top} Jaccard, l*={l_star} vs integrated {lo}-{hi}: {jac:.2f}")
    print(f"  shared: {sorted(A & B)}")

    verdict = {"median_direct_fraction": med, "topk_jaccard": jac,
               "direct_gt_70pct": bool(med > 0.70), "jaccard_lt_30pct": bool(jac < 0.30),
               "reframe_as_readout_writers": bool(med > 0.70 and jac < 0.30)}
    print(f"\n  D14 rule -> direct>70%: {verdict['direct_gt_70pct']}, "
          f"Jaccard<0.30: {verdict['jaccard_lt_30pct']}, "
          f"REFRAME: {verdict['reframe_as_readout_writers']}")
    json.dump({"property": prop, "l_star": l_star, "window": [lo, hi],
               "part_A_direct_split": rowsA, "part_B_integrated_blocks": rowsB,
               "integrated_gap": gap, "verdict": verdict},
              open(f"results/e6_direct_effect_{prop}.json", "w"), indent=1)
    print("wrote results/e6_direct_effect_%s.json" % prop)


if __name__ == "__main__":
    main()
