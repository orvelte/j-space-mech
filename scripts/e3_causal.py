"""E3 — causal validation and the behavioral dissociation (spec §4, H4).

Builds P_prop by adding E2-verified nodes in order of true effect until ablating
the set drives c_prop(l*, p*) to within 30% of the (A) level, capped at 20 nodes.
Ablation is **counterfactual** throughout: activations are replaced with the
same passage's (A)-run values, never zeros.

Mandatory controls (5 seeds each): size- and kind-matched random node sets, and
random attention heads from the broadcast-head band (first half of the workspace
band, spec §4.3.2).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli

NO_THINK_CUE = "\n\nAnswer:<think>\n\n</think>\n\n"


def parse_node(s):
    parts = s.split("/")
    return (parts[0], int(parts[1])) + ((int(parts[2]),) if len(parts) > 2 else ())


def src_values(b, cache_a, node, positions):
    t = (cache_a.attn_in if node[0] == "attn" else
         cache_a.gdn if node[0] == "gdn" else cache_a.mlp)[node[1]][0]
    if node[0] == "attn":
        hd = b.head_dim
        t = t[:, node[2] * hd:(node[2] + 1) * hd]
    return t[positions]


@torch.no_grad()
def coord_at(b, ids, l_star, p_star, target_id, patches=None):
    ctx = hooks.patched(b, patches) if patches else _null()
    with ctx:
        with hooks.NodeCache(b, layers=[l_star]) as c:
            b.model.forward(ids)
            return float(lens_ops.coordinate(b, c.resid[l_star][0, p_star], l_star, target_id))


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


@torch.no_grad()
def generate(b, prompt, n, patches=None, ids=None):
    ids = b.model.encode(prompt) if ids is None else ids
    ctx = hooks.patched(b, patches) if patches else _null()
    with ctx:
        out = b.hf.generate(ids, max_new_tokens=n, do_sample=False,
                            pad_token_id=b.tok.pad_token_id or b.tok.eos_token_id)
    text = b.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()


@torch.no_grad()
def ref_nll(b, ids, ref_ids, patches=None):
    """Cross-entropy of a reference continuation under (possibly ablated) model."""
    full = torch.cat([ids, ref_ids], dim=1)
    ctx = hooks.patched(b, patches) if patches else _null()
    with ctx:
        logits = b.hf(full).logits[0, ids.shape[1] - 1: -1].float()
    return float(torch.nn.functional.cross_entropy(logits, ref_ids[0]))


def random_sets(b, P, l_star, n_seeds, rng, kind="matched"):
    """Size- and kind-matched random node sets, or heads from the broadcast band."""
    allnodes = hooks.nodes_for(b, l_star)
    out = []
    for _ in range(n_seeds):
        if kind == "matched":
            pick = []
            for node in P:
                pool = [n for n in allnodes if n[0] == node[0] and n not in P and n not in pick]
                pick.append(pool[rng.integers(len(pool))])
        else:  # broadcast-head band: first half of the workspace band
            lo, hi = jmodel.WORKSPACE_BAND
            mid = lo + (hi - lo) // 2
            pool = [n for n in allnodes if n[0] == "attn" and lo <= n[1] <= mid]
            if len(pool) < len(P):
                pool = [n for n in allnodes if n[0] == "attn"]
            idx = rng.choice(len(pool), size=min(len(P), len(pool)), replace=False)
            pick = [pool[i] for i in idx]
        out.append(pick)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--prop", default="language")
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-stimuli", type=int, default=None)
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    prop = args.prop
    l_star = jmodel.L_STAR[prop]
    e2 = json.load(open(f"results/e2_attribution_{prop}.json"))
    ranked = [parse_node(d["node"]) for d in e2["verified_nodes"]][:args.cap]
    pairs, rep = stimuli.load(b, prop, max_stimuli=args.max_stimuli)
    tgt = stimuli.target_ids(b, prop)
    target_id = tgt["primary_id"]
    rng = np.random.default_rng(0)
    print(f"{prop}: l*={l_star}, {len(pairs)} stimuli, {len(ranked)} ranked candidates",
          flush=True)

    # cache the (A) activations and clean/corrupt coordinates once per passage
    per = []
    for p in pairs:
        with torch.no_grad():
            cache_a = hooks.NodeCache(b, layers=list(range(l_star + 1)))
            with cache_a:
                b.model.forward(p.ids_a)
        mb = coord_at(b, p.ids_b, l_star, p.p_star, target_id)
        ma = coord_at(b, p.ids_a, l_star, p.p_star, target_id)
        per.append({"pair": p, "cache_a": cache_a, "mb": mb, "ma": ma})
    gap = float(np.mean([d["mb"] - d["ma"] for d in per]))
    print(f"  mean (B)-(A) gap = {gap:.4f}", flush=True)

    def frac_both(ms):
        """(ratio of means, mean of per-passage ratios). The first is primary: the
        per-passage ratio divides by each passage's own gap, which is near zero for
        some passages and makes the mean unstable (it can exceed 100% spuriously)."""
        mbs = [d["mb"] for d in per]; mas = [d["ma"] for d in per]
        denom = float(np.mean(mbs) - np.mean(mas))
        stable = float((np.mean(mbs) - np.mean(ms)) / denom) if denom else float("nan")
        pp = float(np.nanmean([(d["mb"] - m) / (d["mb"] - d["ma"]) if d["mb"] != d["ma"]
                               else np.nan for m, d in zip(ms, per)]))
        return stable, pp

    def ablate_coord(nodes, d):
        p = d["pair"]
        patches = {n: (p.stimulus_positions, src_values(b, d["cache_a"], n, p.stimulus_positions))
                   for n in nodes}
        return coord_at(b, p.ids_b, l_star, p.p_star, target_id, patches)

    # ---- greedy prefix curve -------------------------------------------
    curve = []
    for k in range(1, len(ranked) + 1):
        ms = [ablate_coord(ranked[:k], d) for d in per]
        frac, frac_pp = frac_both(ms)
        curve.append({"k": k, "mean_coord": float(np.mean(ms)), "frac_of_gap_closed": frac,
                      "frac_per_passage_estimator": frac_pp})
        print(f"  k={k:>2} {ranked[k-1]} -> {frac:.1%} of the gap closed", flush=True)
        if frac >= 0.70:
            break
    achieved = curve[-1]["frac_of_gap_closed"]
    reached = achieved >= 0.70
    P = ranked[:curve[-1]["k"]]
    note = "" if reached else f"  (cap reached; only {achieved:.1%} of the gap closed)"
    print(f"  |P_{prop}| = {len(P)}{note}", flush=True)

    # ---- mandatory random baselines ------------------------------------
    baselines = {}
    for kind in ("matched", "broadcast_band"):
        fr = []
        for s, nodes in enumerate(random_sets(b, P, l_star, args.seeds, rng, kind)):
            ms = [ablate_coord(nodes, d) for d in per]
            fr.append(frac_both(ms)[0])
            print(f"  random[{kind}] seed {s}: {fr[-1]:.1%}", flush=True)
        baselines[kind] = {"per_seed": fr, "mean": float(np.mean(fr)),
                           "sd": float(np.std(fr, ddof=1))}

    # ---- H4 behavioral dissociation -------------------------------------
    named_base, named_abl, nll_base, nll_abl = [], [], [], []
    lang_ok, lang_ok_base = [], []
    for d in per:
        p = d["pair"]
        patches = {n: (p.stimulus_positions, src_values(b, d["cache_a"], n, p.stimulus_positions))
                   for n in P}
        exp = [w.strip().lower() for w in tgt["expected"]]
        for store, pat in ((named_base, None), (named_abl, patches)):
            ans = generate(b, p.prompt_b + NO_THINK_CUE, 16, pat)
            store.append(any(w in ans.lower() for w in exp))
        ref = generate(b, p.prompt_a, 10)
        ref_ids = b.tok(ref, return_tensors="pt").input_ids.cuda()
        if ref_ids.shape[1]:
            nll_base.append(ref_nll(b, p.ids_a, ref_ids))
            nll_abl.append(ref_nll(b, p.ids_a, ref_ids, patches))
        if prop == "language":
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            # Spec E3 names langdetect as a behavioural metric for language, so the
            # *baseline* rate is required too -- an ablated rate alone cannot show
            # whether behaviour changed.
            for store, pat in ((lang_ok_base, None), (lang_ok, patches)):
                cont = generate(b, p.prompt_a, 20, pat)
                try:
                    store.append(detect(cont) == "es")
                except Exception:
                    store.append(False)

    acc_b, acc_a = float(np.mean(named_base)), float(np.mean(named_abl))
    ppl_b, ppl_a = float(np.exp(np.mean(nll_base))), float(np.exp(np.mean(nll_abl)))
    coord_drop = achieved
    behav_ok = (abs(acc_a - acc_b) <= 0.10 * max(acc_b, 1e-9)) and (abs(ppl_a - ppl_b) <= 0.10 * ppl_b)
    lang_b = float(np.mean(lang_ok_base)) if lang_ok_base else None
    lang_a = float(np.mean(lang_ok)) if lang_ok else None
    if lang_b is not None:
        # Third behavioural metric, language only: is the continuation still Spanish?
        behav_ok = behav_ok and abs(lang_a - lang_b) <= 0.10 * max(lang_b, 1e-9)

    out = {
        "property": prop, "l_star": l_star, "n_stimuli": len(pairs),
        "BA_gap_mean": gap, "curve": curve,
        "P_prop": ["/".join(str(x) for x in n) for n in P], "size": len(P),
        "reached_70pct": bool(reached),
        "frac_of_gap_closed": coord_drop,
        "K4_diffuse": bool(not reached and baselines["matched"]["mean"] >= 0.5 * coord_drop),
        "random_baselines": baselines,
        "H4": {"naming_accuracy_baseline": acc_b, "naming_accuracy_ablated": acc_a,
               "ref_continuation_ppl_baseline": ppl_b, "ref_continuation_ppl_ablated": ppl_a,
               "still_spanish_baseline": lang_b, "still_spanish_ablated": lang_a,
               "behaviour_within_10pct": bool(behav_ok),
               "dissociation_holds": bool(coord_drop >= 0.70 and behav_ok)},
    }
    json.dump(out, open(f"results/e3_causal_{prop}.json", "w"), indent=1)
    print(f"\n|P|={len(P)} closes {coord_drop:.1%} of the gap; "
          f"random matched {baselines['matched']['mean']:.1%}, "
          f"broadcast-band heads {baselines['broadcast_band']['mean']:.1%}")
    print(f"H4: naming {acc_b:.0%} -> {acc_a:.0%}, ref-continuation ppl "
          f"{ppl_b:.2f} -> {ppl_a:.2f}, dissociation {out['H4']['dissociation_holds']}")


if __name__ == "__main__":
    main()
