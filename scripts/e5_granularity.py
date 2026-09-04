"""Granularity fix: compare GDN, MLP and attention at matched block level (D13).

E2 ranked single attention heads against whole GDN/MLP blocks. This re-does both
the mass breakdown and the promoter ranking with attention aggregated per layer.

Mass is recomputed from the cached E2 arrays -- a layer's block attribution is the
sum of its per-head attributions, since those are inner products over disjoint
slices of one tensor. True effects are measured fresh by patching whole blocks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from jspace import hooks, model as jmodel, stimuli
from scripts.e3_causal import coord_at, src_values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--props", default="language,tense,pos")
    ap.add_argument("--out", default="results/e5_granularity.json")
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    out = {}
    for prop in args.props.split(","):
        l_star = jmodel.L_STAR[prop]
        z = np.load(f"results/e2_attribution_{prop}_raw.npz")
        nodes = [str(x) for x in z["nodes"]]

        # ---- block-level attribution mass, from cached arrays ------------
        blocks = defaultdict(lambda: {"stim": 0.0, "ques": 0.0})
        for n, ms, mq in zip(nodes, z["mean_stim"], z["mean_ques"]):
            parts = n.split("/")
            key = (parts[0], int(parts[1]))          # heads collapse into their layer
            blocks[key]["stim"] += float(ms)          # signed sum == block attribution
            blocks[key]["ques"] += float(mq)

        def mass(kind, field):
            return sum(abs(v[field]) for k, v in blocks.items() if k[0] == kind)

        kinds = ("attn", "gdn", "mlp")
        total = sum(mass(k, f) for k in kinds for f in ("stim", "ques"))
        block_mass = {f"{k}_at_{'stimulus' if f == 'stim' else 'question'}":
                      mass(k, f) / total for k in kinds for f in ("stim", "ques")}

        # the old, unmatched numbers for comparison
        old_total = float(np.abs(z["mean_stim"]).sum() + np.abs(z["mean_ques"]).sum())
        def old_mass(kind, arr):
            return float(sum(abs(a) for n, a in zip(nodes, arr) if n.split("/")[0] == kind))
        old = {f"{k}_at_{'stimulus' if f == 'stim' else 'question'}":
               old_mass(k, z["mean_stim" if f == "stim" else "mean_ques"]) / old_total
               for k in kinds for f in ("stim", "ques")}

        print(f"\n=== {prop} (l*={l_star}) — attribution mass ===")
        print(f"{'':<22}{'per-head (E2, unfair)':>22}{'block-level (matched)':>23}")
        for k in ("attn", "gdn", "mlp"):
            for f in ("stimulus", "question"):
                key = f"{k}_at_{f}"
                print(f"  {key:<20}{old[key]:>21.1%}{block_mass[key]:>23.1%}")

        # ---- true block-level effects, measured by patching --------------
        pairs, _ = stimuli.load(b, prop)
        tgt = stimuli.target_ids(b, prop)["primary_id"]
        per = []
        for p in pairs:
            with torch.no_grad():
                ca = hooks.NodeCache(b, layers=list(range(l_star + 1)))
                with ca:
                    b.model.forward(p.ids_a)
            per.append({"pair": p, "cache_a": ca,
                        "mb": coord_at(b, p.ids_b, l_star, p.p_star, tgt),
                        "ma": coord_at(b, p.ids_a, l_star, p.p_star, tgt)})
        mb_mean = float(np.mean([d["mb"] for d in per]))
        gap = mb_mean - float(np.mean([d["ma"] for d in per]))

        block_list = []
        for l in range(l_star + 1):
            block_list.append(("attn", l) if b.is_full_attn(l) else ("gdn", l))
            block_list.append(("mlp", l))

        rows = []
        for bl in block_list:
            kind, l = bl
            members = ([("attn", l, h) for h in range(b.n_heads)] if kind == "attn"
                       else [bl])
            ms = []
            for d in per:
                p = d["pair"]
                patches = {n: (p.stimulus_positions,
                               src_values(b, d["cache_a"], n, p.stimulus_positions))
                           for n in members}
                ms.append(coord_at(b, p.ids_b, l_star, p.p_star, tgt, patches))
            delta = float(np.mean(ms)) - mb_mean
            rows.append({"block": f"{kind}/{l}", "kind": kind, "layer": l,
                         "true_delta_m": delta, "frac_of_gap": delta / gap if gap else None,
                         "attribution": blocks[bl]["stim"]})
            print(f"  patched {kind}/{l:<3} Δm {delta:>8.4f}  "
                  f"({delta / gap:>6.0%} of gap)", flush=True)

        rows.sort(key=lambda r: r["true_delta_m"])
        print(f"\n  top 10 blocks, {prop}:")
        for r in rows[:10]:
            print(f"    {r['block']:<10} Δm {r['true_delta_m']:>8.4f}  {r['frac_of_gap']:>7.0%}")
        top10 = rows[:10]
        n_attn = sum(1 for r in top10 if r["kind"] == "attn")
        print(f"  attention blocks in the top 10: {n_attn}/10")

        out[prop] = {"l_star": l_star, "BA_gap_mean": gap,
                     "mass_block_level": block_mass, "mass_per_head_E2": old,
                     "blocks": rows,
                     "attn_blocks_in_top10": n_attn,
                     "hybrid_claim_survives": bool(
                         block_mass["gdn_at_stimulus"] > block_mass["attn_at_stimulus"]
                         and n_attn < 5)}
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
