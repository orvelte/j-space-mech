"""E4b — is `gdn/24` a real shared gate, or one representative of many? (§15.2)

E3's greedy prefix returns *a* sufficient set, not *the* promotion set. Overlap
and cross-ablation between single representatives therefore compare arbitrary
members of a family and understate any true sharing. This experiment samples the
family and asks three questions the single-representative analysis cannot:

1. **Frequency.** Across many independently sampled sufficient sets, how often
   does each node appear? A node in nearly every set is structurally required;
   one appearing occasionally is interchangeable.
2. **Necessity.** Re-run the search with `gdn/24` *banned*. If sufficient sets
   still exist without it, it is not a gate — it is a convenient member.
   This is the decisive test.
3. **Core overlap.** Compare each property's *core* (nodes above a frequency
   threshold) rather than one representative, against a matched random baseline.

Search: randomized greedy over the E2-verified candidates. Each step samples
uniformly from the top-`--width` remaining candidates by true effect, so restarts
explore different sets. Sets are discovered on a fixed subsample of passages for
speed, then **re-validated on all passages** before being counted as sufficient.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter

import numpy as np
import torch

from jspace import hooks, model as jmodel, stimuli
from scripts.e3_causal import coord_at, parse_node, src_values

PROPS = ["language", "tense", "pos"]
TARGET = 0.70


def build_cache(b, prop, l_star):
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
    return per, tgt


def frac_closed(b, nodes, per, l_star, tgt, idx=None, both=False):
    """Fraction of the (B)-(A) gap closed by ablating `nodes`.

    Primary estimator is the **ratio of means**, not the mean of per-passage
    ratios. The per-passage ratio has (mb - ma) in the denominator, which is near
    zero for some passages, so its mean is dominated by those passages and can
    report absurd values (a smoke run produced "357%"). The ratio of means is
    bounded by the aggregate effect and is what is reported here; the unstable
    per-passage mean is kept alongside it for comparison with E3.
    """
    sel = per if idx is None else [per[i] for i in idx]
    ms, mbs, mas = [], [], []
    for d in sel:
        p = d["pair"]
        patches = {n: (p.stimulus_positions,
                       src_values(b, d["cache_a"], n, p.stimulus_positions)) for n in nodes}
        ms.append(coord_at(b, p.ids_b, l_star, p.p_star, tgt, patches))
        mbs.append(d["mb"]); mas.append(d["ma"])
    denom = float(np.mean(mbs) - np.mean(mas))
    stable = float((np.mean(mbs) - np.mean(ms)) / denom) if denom else float("nan")
    if not both:
        return stable
    per_passage = float(np.nanmean([(b_ - m) / (b_ - a) if b_ != a else np.nan
                                    for m, b_, a in zip(ms, mbs, mas)]))
    return stable, per_passage


def search(b, cands, per, l_star, tgt, rng, *, width, cap, search_idx):
    """One randomized-greedy run; returns (set, frac on the search subsample)."""
    chosen, pool = [], list(cands)
    for _ in range(cap):
        if not pool:
            break
        chosen.append(pool.pop(rng.integers(min(width, len(pool)))))
        f = frac_closed(b, chosen, per, l_star, tgt, search_idx)
        if f >= TARGET:
            return chosen, f
    return chosen, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--runs", type=int, default=18)
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--search-n", type=int, default=8)
    ap.add_argument("--props", default="language,tense,pos")
    ap.add_argument("--out", default="results/e4b_set_family.json")
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    rng = np.random.default_rng(0)
    GATE = ("gdn", 24)
    out = {}

    for prop in args.props.split(","):
        l_star = jmodel.L_STAR[prop]
        cands = [parse_node(d["node"]) for d in
                 json.load(open(f"results/e2_attribution_{prop}.json"))["verified_nodes"]]
        per, tgt = build_cache(b, prop, l_star)
        search_idx = list(range(min(args.search_n, len(per))))
        gate_available = GATE in cands
        print(f"\n=== {prop}  l*={l_star}  {len(per)} passages, {len(cands)} candidates"
              f"  (gdn/24 among them: {gate_available})", flush=True)

        results = {}
        for mode in ("free", "no_gate"):
            pool = cands if mode == "free" else [n for n in cands if n != GATE]
            sets = []
            for r in range(args.runs):
                s, f_search = search(b, pool, per, l_star, tgt, rng,
                                     width=args.width, cap=args.cap, search_idx=search_idx)
                f_all, f_pp = frac_closed(b, s, per, l_star, tgt, both=True)  # re-validate on all
                sets.append({"nodes": ["/".join(str(x) for x in n) for n in s],
                             "size": len(s), "frac_search": f_search, "frac_all": f_all,
                             "frac_all_per_passage_estimator": f_pp,
                             "sufficient": bool(f_all >= TARGET)})
                print(f"  [{mode}] run {r+1}/{args.runs}: |S|={len(s)} "
                      f"search {f_search:.0%} / all {f_all:.0%} "
                      f"{'OK' if f_all >= TARGET else '--'}", flush=True)
            suff = [s for s in sets if s["sufficient"]]
            freq = Counter(n for s in suff for n in s["nodes"])
            results[mode] = {
                "n_runs": len(sets), "n_sufficient": len(suff),
                "mean_size": float(np.mean([s["size"] for s in suff])) if suff else None,
                "mean_frac_all": float(np.mean([s["frac_all"] for s in suff])) if suff else None,
                "node_frequency": {k: v / len(suff) for k, v in freq.most_common()} if suff else {},
                "sets": sets,
            }
        f_free = results["free"]["node_frequency"].get("gdn/24", 0.0)
        results["gate_verdict"] = {
            "gdn24_frequency_when_available": f_free,
            "sufficient_sets_without_gdn24": results["no_gate"]["n_sufficient"],
            "size_penalty_without_gdn24": (
                (results["no_gate"]["mean_size"] or float("nan")) -
                (results["free"]["mean_size"] or float("nan"))),
            "gdn24_is_necessary": bool(results["no_gate"]["n_sufficient"] == 0),
        }
        print(f"  -> gdn/24 appears in {f_free:.0%} of free sufficient sets; "
              f"{results['no_gate']['n_sufficient']}/{args.runs} sufficient sets exist "
              f"without it", flush=True)
        out[prop] = results

    # ---- core overlap across properties ---------------------------------
    def core(prop, thresh):
        return {k for k, v in out[prop]["free"]["node_frequency"].items() if v >= thresh}

    overlap = {}
    for thresh in (0.5, 0.75):
        rows = []
        for a, c in itertools.combinations([p for p in out], 2):
            L = min(jmodel.L_STAR[a], jmodel.L_STAR[c])
            A = {n for n in core(a, thresh) if int(n.split("/")[1]) <= L}
            B = {n for n in core(c, thresh) if int(n.split("/")[1]) <= L}
            j = len(A & B) / len(A | B) if (A | B) else float("nan")
            rows.append({"a": a, "b": c, "threshold": thresh, "core_a": sorted(A),
                         "core_b": sorted(B), "shared": sorted(A & B), "jaccard": j})
            print(f"core@{thresh:.2f} {a} vs {c}: |A|={len(A)} |B|={len(B)} "
                  f"shared={sorted(A & B)} J={j:.2f}")
        overlap[str(thresh)] = rows
    out["core_overlap"] = overlap
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
