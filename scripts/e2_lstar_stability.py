"""Robustness of the E2 promoter set to the choice of l* (DECISIONS.md D10).

Tense's l* was moved 21 -> 28 post-hoc. If the verified promoter set is stable
across 27 / 28 / 29 -- adjacent layers inside the same >2 SE run -- then the
specific choice of 28 carries little weight and the post-hoc concern is largely
defused. If it is not stable, the tense result depends on a hand-picked layer and
must be reported that way.
"""

from __future__ import annotations

import json
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {27: "results/e2_attribution_tense_L27.json",
        28: "results/e2_attribution_tense.json",
        29: "results/e2_attribution_tense_L29.json"}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def main():
    data = {l: json.load(open(p)) for l, p in RUNS.items() if __import__("os").path.exists(p)}
    if len(data) < 2:
        print("need at least two runs; have", sorted(data)); return
    ls = sorted(data)
    verified = {l: [d["node"] for d in data[l]["verified_nodes"]] for l in ls}
    effects = {l: {d["node"]: d["true_delta_m"] for d in data[l]["verified_nodes"]} for l in ls}

    print("per-run summary")
    for l in ls:
        d = data[l]
        print(f"  l*={l}: gap {d['BA_gap_mean']:.3f}, r={d['attribution_vs_patching_corr']:.3f}, "
              f"verified {d['n_verified']}/{d['n_candidates']}")

    print("\npairwise stability")
    rows = []
    for a, b in itertools.combinations(ls, 2):
        j = jaccard(verified[a], verified[b])
        common = sorted(set(verified[a]) & set(verified[b]))
        if len(common) >= 3:
            x = [effects[a][n] for n in common]
            y = [effects[b][n] for n in common]
            from scipy.stats import spearmanr
            rho = float(spearmanr(x, y).statistic)
        else:
            rho = float("nan")
        top5 = jaccard(verified[a][:5], verified[b][:5])
        rows.append((a, b, j, rho, len(common), top5))
        print(f"  l*={a} vs l*={b}: Jaccard {j:.2f} | top-5 Jaccard {top5:.2f} | "
              f"{len(common)} shared | Spearman rho of effects {rho:.2f}")

    print("\ntop 8 promoters per run")
    width = max(len(str(l)) for l in ls)
    for l in ls:
        print(f"  l*={l:<{width}}: {', '.join(verified[l][:8])}")

    print("\nattribution mass by node type")
    for l in ls:
        m = data[l]["attribution_mass"]
        print(f"  l*={l}: " + "  ".join(
            f"{k.replace('_at_',' @ '):<18}{v:.0%}" for k, v in m.items()))

    stable = all(r[2] >= 0.5 for r in rows)
    print(f"\nverdict: promoter set is {'STABLE' if stable else 'NOT stable'} "
          f"across l* (all pairwise Jaccard >= 0.5)" if rows else "")

    out = {"runs": {str(l): {"l_star": l, "verified": verified[l],
                             "corr": data[l]["attribution_vs_patching_corr"],
                             "mass": data[l]["attribution_mass"]} for l in ls},
           "pairwise": [{"a": a, "b": b, "jaccard": j, "top5_jaccard": t5,
                         "n_shared": n, "spearman_rho": rho}
                        for a, b, j, rho, n, t5 in rows],
           "stable": bool(stable)}
    json.dump(out, open("results/e2_tense_lstar_stability.json", "w"), indent=1)

    # figure: which nodes appear at which l*, ordered by effect at l*=28
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    anchor = 28 if 28 in ls else ls[0]
    order = verified[anchor][:16]
    order += [n for n in set().union(*verified.values()) if n not in order][:6]
    y = np.arange(len(order))
    for k, l in enumerate(ls):
        xs = [effects[l].get(n, np.nan) for n in order]
        ax.scatter([-x if x == x else np.nan for x in xs], y,
                   s=30, label=f"l* = {l}", alpha=.85,
                   marker=["o", "s", "^"][k % 3])
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("drop in J-lens coordinate when ablated (−Δm)")
    ax.legend(fontsize=7.5)
    ax.set_title("E2 robustness: tense promoters across adjacent l*\n"
                 "(missing marker = not verified at that l*)", fontsize=9)
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig("figures/e2_tense_lstar_stability.png"); plt.close(fig)
    print("wrote results/e2_tense_lstar_stability.json + figures/e2_tense_lstar_stability.png")


if __name__ == "__main__":
    main()
