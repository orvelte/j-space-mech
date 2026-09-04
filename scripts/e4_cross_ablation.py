"""E4 — cross-ablation transfer matrix (spec §4, H3's second half).

Ablate P_i and measure the coordinate drop for property j, normalised by the
within-property drop. Off-diagonals near 1 = a shared gate; near 0 =
property-specific machinery.

**A structural caveat this design has to state.** l* differs per property
(language 24, tense 28, pos 30), so a node in P_i at a layer above l*_j cannot
influence the measurement at l*_j at all -- it is downstream of the readout. Such
nodes are dropped from the cross-ablation and counted in `n_dropped_above_lstar`.
An off-diagonal is therefore a transfer estimate for the *applicable* part of
P_i, and a low value caused mostly by dropped nodes is not evidence against a
shared gate. Both numbers are reported.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, model as jmodel, stimuli
from scripts.e3_causal import coord_at, parse_node, src_values

PROPS = ["language", "tense", "pos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    args = ap.parse_args()
    b = jmodel.load(args.variant)

    P = {p: [parse_node(n) for n in json.load(open(f"results/e3_causal_{p}.json"))["P_prop"]]
         for p in PROPS}
    for p in PROPS:
        print(f"P_{p} ({len(P[p])}): {[('/'.join(str(x) for x in n)) for n in P[p]]}")

    matrix, detail = {}, {}
    for j in PROPS:                      # measured property
        l_star = jmodel.L_STAR[j]
        pairs, _ = stimuli.load(b, j)
        tgt = stimuli.target_ids(b, j)["primary_id"]
        per = []
        for p in pairs:
            with torch.no_grad():
                cache_a = hooks.NodeCache(b, layers=list(range(l_star + 1)))
                with cache_a:
                    b.model.forward(p.ids_a)
            per.append({"pair": p, "cache_a": cache_a,
                        "mb": coord_at(b, p.ids_b, l_star, p.p_star, tgt),
                        "ma": coord_at(b, p.ids_a, l_star, p.p_star, tgt)})

        drops = {}
        for i in PROPS:                  # ablated property's set
            applicable = [n for n in P[i] if n[1] <= l_star]
            dropped = len(P[i]) - len(applicable)
            if not applicable:
                drops[i] = {"frac_of_gap": 0.0, "n_applied": 0,
                            "n_dropped_above_lstar": dropped}
                continue
            fr = []
            for d in per:
                pr = d["pair"]
                patches = {n: (pr.stimulus_positions,
                               src_values(b, d["cache_a"], n, pr.stimulus_positions))
                           for n in applicable}
                m = coord_at(b, pr.ids_b, l_star, pr.p_star, tgt, patches)
                fr.append((d["mb"] - m) / (d["mb"] - d["ma"]) if d["mb"] != d["ma"] else np.nan)
            drops[i] = {"frac_of_gap": float(np.nanmean(fr)),
                        "se": float(np.nanstd(fr, ddof=1) / np.sqrt(len(fr))),
                        "n_applied": len(applicable),
                        "n_dropped_above_lstar": dropped}
            print(f"  ablate P_{i:<9} measure {j:<9} -> {drops[i]['frac_of_gap']:>7.1%} "
                  f"({len(applicable)}/{len(P[i])} nodes applicable)", flush=True)
        within = drops[j]["frac_of_gap"]
        matrix[j] = {i: (drops[i]["frac_of_gap"] / within if within else float("nan"))
                     for i in PROPS}
        detail[j] = drops

    print("\ntransfer matrix (rows = measured property, normalised by within-property drop)")
    print(f"{'measured':<10}" + "".join(f"{'P_'+i:>12}" for i in PROPS))
    for j in PROPS:
        print(f"{j:<10}" + "".join(f"{matrix[j][i]:>11.2f}" for i in PROPS))

    off = [matrix[j][i] for j in PROPS for i in PROPS if i != j]
    verdict = {"mean_offdiagonal": float(np.mean(off)),
               "H3_transfer_>=50pct": bool(np.mean(off) >= 0.50),
               "H3_falsified_<20pct": bool(np.mean(off) < 0.20)}
    print(f"\nmean off-diagonal transfer {verdict['mean_offdiagonal']:.1%} "
          f"(H3 predicts >=50%; <20% falsifies)")
    json.dump({"matrix": matrix, "detail": detail, "verdict": verdict,
               "P_prop": {p: ["/".join(str(x) for x in n) for n in P[p]] for p in PROPS}},
              open("results/e4_cross_ablation.json", "w"), indent=1)
    print("wrote results/e4_cross_ablation.json")


if __name__ == "__main__":
    main()
