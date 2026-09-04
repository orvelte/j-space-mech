"""H3 preview: do the properties share promoter nodes? (spec §4 E4, overlap half)

The full E4 also needs the cross-ablation matrix; this is the overlap statistic
alone, computed from the E2 outputs.

**Pools differ across properties** because l* differs (language 24, tense 28,
pos 30): language cannot have a promoter at layer 25. Every pairwise comparison
is therefore restricted to the common pool -- nodes at layers <= min(l*_a, l*_b)
-- and the random baseline is drawn from that same restricted pool. Comparing
raw Jaccards across different pools would be meaningless.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from jspace.model import L_STAR

RUNS = {"language": "results/e2_attribution_language.json",
        "tense": "results/e2_attribution_tense.json",
        "pos": "results/e2_attribution_pos.json"}
N_HEADS = 24


def pool(max_layer):
    """Every node at layers <= max_layer, matching hooks.nodes_for."""
    out = []
    for l in range(max_layer + 1):
        if l % 4 == 3:
            out += [f"attn/{l}/{h}" for h in range(N_HEADS)]
        else:
            out.append(f"gdn/{l}")
        out.append(f"mlp/{l}")
    return out


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def main(n_draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    data = {k: json.load(open(v)) for k, v in RUNS.items()}
    verified = {k: [d["node"] for d in v["verified_nodes"]] for k, v in data.items()}
    rows = []
    print(f"{'pair':<22} {'pool<=L':>8} {'|A|':>4} {'|B|':>4} {'shared':>7} "
          f"{'Jaccard':>8} {'chance':>8} {'ratio':>7} {'p':>7}")
    for a, b in itertools.combinations(RUNS, 2):
        L = min(L_STAR[a], L_STAR[b])
        P = pool(L)
        A = [n for n in verified[a] if n in set(P)]
        B = [n for n in verified[b] if n in set(P)]
        obs = jac(A, B)
        null = np.array([jac(rng.choice(len(P), len(A), replace=False),
                             rng.choice(len(P), len(B), replace=False))
                         for _ in range(n_draws)])
        chance = float(null.mean())
        p = float((null >= obs).mean())
        ratio = obs / chance if chance > 0 else float("inf")
        print(f"{a+' vs '+b:<22} {L:>8} {len(A):>4} {len(B):>4} "
              f"{len(set(A)&set(B)):>7} {obs:>8.3f} {chance:>8.3f} {ratio:>7.1f} {p:>7.3f}")
        rows.append({"a": a, "b": b, "common_pool_max_layer": L, "pool_size": len(P),
                     "n_a": len(A), "n_b": len(B), "shared": sorted(set(A) & set(B)),
                     "jaccard": obs, "chance_jaccard": chance,
                     "ratio_vs_chance": ratio, "p_value": p,
                     "meets_3x_prediction": bool(ratio >= 3.0)})

    common3 = set.intersection(*[set(v) for v in verified.values()])
    print(f"\nshared by all three properties ({len(common3)}): {sorted(common3)}")
    for n in sorted(common3):
        eff = {k: next((d['frac_of_BA_gap'] for d in data[k]['verified_nodes']
                        if d['node'] == n), None) for k in RUNS}
        print(f"   {n:<12} " + "  ".join(f"{k} {v:.0%}" for k, v in eff.items()))
    json.dump({"pairwise": rows, "shared_by_all_three": sorted(common3)},
              open("results/e4_overlap_preview.json", "w"), indent=1)
    print("\nwrote results/e4_overlap_preview.json")


if __name__ == "__main__":
    main()
