"""E2 — attribution over nodes, verified by activation patching (spec §4).

Metric m = c_prop(l*, p*). Clean run = condition (B); corrupted run = (A).
Attribution patching estimates, for every node, Dm ~ <grad m, a_corrupt - a_clean>.

Sign convention: (B) holds the higher coordinate, so replacing a *promoter*'s
clean activation with its corrupted one should *lower* m. **Promoters have
negative Dm** and are ranked most-negative-first throughout.

Node granularity (DECISIONS D9): a node is (kind, layer, head) intervened at all
stimulus positions at once, not one (node, position) pair per the spec's literal
wording. Per-position attribution is still computed and is what the H1-vs-H2
breakdown in `attribution_mass` reports; the all-stimulus-positions intervention
is what verification and E3's ablation actually apply, so ranking on it keeps
the estimate and the verification measuring the same thing.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jspace import hooks, lens_ops, model as jmodel, stimuli


def node_key(node):
    return "/".join(str(x) for x in node)


def capture(b, ids, layers, *, build_graph, patterns=False):
    cache = hooks.NodeCache(b, layers=layers, build_graph=build_graph,
                            attn_patterns=patterns)
    with cache:
        b.model.forward(ids)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="27b")
    ap.add_argument("--prop", default="language")
    ap.add_argument("--l-star", type=int, default=None,
                    help="defaults to jspace.model.L_STAR[prop]")
    ap.add_argument("--max-stimuli", type=int, default=None)
    ap.add_argument("--top-heads", type=int, default=30)
    ap.add_argument("--top-mlps", type=int, default=8)
    ap.add_argument("--top-gdn", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b = jmodel.load(args.variant)
    l_star = args.l_star if args.l_star is not None else jmodel.L_STAR[args.prop]
    layers = list(range(l_star + 1))
    pairs, rep = stimuli.load(b, args.prop, max_stimuli=args.max_stimuli)
    tgt = stimuli.target_ids(b, args.prop)
    target_id = tgt["primary_id"]
    nodes = hooks.nodes_for(b, l_star)
    print(f"{args.prop}: l*={l_star}, {len(pairs)} stimuli "
          f"({rep['n_excluded_by_e02']} excluded by E0.2), {len(nodes)} nodes "
          f"({sum(1 for n in nodes if n[0]=='attn')} heads, "
          f"{sum(1 for n in nodes if n[0]=='gdn')} gdn, "
          f"{sum(1 for n in nodes if n[0]=='mlp')} mlp)", flush=True)

    # ---- attribution patching -------------------------------------------
    attr_stim = np.zeros((len(pairs), len(nodes)), dtype=np.float64)
    attr_ques = np.zeros((len(pairs), len(nodes)), dtype=np.float64)
    m_clean = np.zeros(len(pairs))
    m_corrupt = np.zeros(len(pairs))

    for pi, p in enumerate(pairs):
        with torch.no_grad():
            cache_a = capture(b, p.ids_a, layers + [l_star], build_graph=False)
            m_corrupt[pi] = float(
                lens_ops.coordinate(b, cache_a.resid[l_star][0, p.p_star], l_star, target_id))

        cache_b = hooks.NodeCache(b, layers=layers, build_graph=True)
        with cache_b:
            b.model.forward(p.ids_b)
            m = lens_ops.coordinate(b, cache_b.resid[l_star][0, p.p_star], l_star, target_id)
            tensors, index = [], {}
            for l in layers:
                if b.is_full_attn(l):
                    index[("attn", l)] = len(tensors); tensors.append(cache_b.attn_in[l])
                else:
                    index[("gdn", l)] = len(tensors); tensors.append(cache_b.gdn[l])
                index[("mlp", l)] = len(tensors); tensors.append(cache_b.mlp[l])
            grads = torch.autograd.grad(m, tensors)
        m_clean[pi] = float(m)

        stim = torch.tensor(p.stimulus_positions, device="cuda")
        ques = torch.arange(0, p.stimulus_start, device="cuda")
        for ni, node in enumerate(nodes):
            gi = index[(node[0], node[1])]
            g = grads[gi][0]                                   # [T, width]
            clean = (cache_b.attn_in if node[0] == "attn" else
                     cache_b.gdn if node[0] == "gdn" else cache_b.mlp)[node[1]][0].detach()
            corrupt = (cache_a.attn_in if node[0] == "attn" else
                       cache_a.gdn if node[0] == "gdn" else cache_a.mlp)[node[1]][0]
            if node[0] == "attn":
                hd = b.head_dim; sl = slice(node[2] * hd, (node[2] + 1) * hd)
                g, clean, corrupt = g[:, sl], clean[:, sl], corrupt[:, sl]
            delta = (corrupt.float() - clean.float()) * g.float()   # [T, width]
            per_pos = delta.sum(-1)
            attr_stim[pi, ni] = float(per_pos[stim].sum())
            attr_ques[pi, ni] = float(per_pos[ques].sum())
        del cache_b, grads
        torch.cuda.empty_cache()
        print(f"  attribution [{pi+1}/{len(pairs)}] m(B)={m_clean[pi]:.3f} "
              f"m(A)={m_corrupt[pi]:.3f}", flush=True)

    mean_stim = attr_stim.mean(0)
    mean_ques = attr_ques.mean(0)

    # H1 vs H2: where does the attribution mass sit?
    def mass(kinds, arr):
        return float(sum(abs(arr[i]) for i, n in enumerate(nodes) if n[0] in kinds))
    total = mass(("attn", "gdn", "mlp"), mean_stim) + mass(("attn", "gdn", "mlp"), mean_ques)
    breakdown = {
        "attn_at_stimulus": mass(("attn",), mean_stim) / total,
        "attn_at_question": mass(("attn",), mean_ques) / total,
        "gdn_at_stimulus": mass(("gdn",), mean_stim) / total,
        "gdn_at_question": mass(("gdn",), mean_ques) / total,
        "mlp_at_stimulus": mass(("mlp",), mean_stim) / total,
        "mlp_at_question": mass(("mlp",), mean_ques) / total,
    }

    # ---- candidates: most-negative mean attribution per kind -------------
    order = np.argsort(mean_stim)  # most negative first
    picks, counts = [], {"attn": 0, "mlp": 0, "gdn": 0}
    caps = {"attn": args.top_heads, "mlp": args.top_mlps, "gdn": args.top_gdn}
    for i in order:
        k = nodes[i][0]
        if counts[k] < caps[k]:
            picks.append(int(i)); counts[k] += 1
    print(f"\nverifying {len(picks)} candidates by activation patching "
          f"({counts})", flush=True)

    # ---- verification by activation patching ----------------------------
    true_delta = np.zeros((len(pairs), len(picks)))
    for pi, p in enumerate(pairs):
        with torch.no_grad():
            cache_a = capture(b, p.ids_a, layers, build_graph=False)
            for ci, i in enumerate(picks):
                node = nodes[i]
                src = (cache_a.attn_in if node[0] == "attn" else
                       cache_a.gdn if node[0] == "gdn" else cache_a.mlp)[node[1]][0]
                if node[0] == "attn":
                    hd = b.head_dim
                    src = src[:, node[2] * hd:(node[2] + 1) * hd]
                vals = src[p.stimulus_positions]
                with hooks.patched(b, {node: (p.stimulus_positions, vals)}):
                    with hooks.NodeCache(b, layers=[l_star]) as c:
                        b.model.forward(p.ids_b)
                        mm = float(lens_ops.coordinate(
                            b, c.resid[l_star][0, p.p_star], l_star, target_id))
                true_delta[pi, ci] = mm - m_clean[pi]
        print(f"  patching [{pi+1}/{len(pairs)}]", flush=True)

    est = mean_stim[picks]
    true_mean = true_delta.mean(0)
    corr = float(np.corrcoef(est, true_mean)[0, 1])
    keep = [(ci, i) for ci, i in enumerate(picks)
            if true_mean[ci] < 0 and abs(true_mean[ci]) >= 0.25 * abs(est[ci])]

    verified = sorted(
        ({"node": node_key(nodes[i]), "kind": nodes[i][0], "layer": nodes[i][1],
          "head": nodes[i][2] if nodes[i][0] == "attn" else None,
          "attribution_estimate": float(est[ci]), "true_delta_m": float(true_mean[ci]),
          "true_delta_se": float(true_delta[:, ci].std(ddof=1) / np.sqrt(len(pairs))),
          "frac_of_BA_gap": float(true_mean[ci] / (m_clean - m_corrupt).mean())}
         for ci, i in keep), key=lambda d: d["true_delta_m"])

    out = {
        "property": args.prop, "l_star": l_star, "n_stimuli": len(pairs),
        "n_excluded_by_e02": rep["n_excluded_by_e02"],
        "m_clean_mean": float(m_clean.mean()), "m_corrupt_mean": float(m_corrupt.mean()),
        "BA_gap_mean": float((m_clean - m_corrupt).mean()),
        "attribution_mass": breakdown,
        "attribution_vs_patching_corr": corr,
        "K3_fires_below_0.4": bool(corr < 0.4),
        "n_candidates": len(picks), "n_verified": len(verified),
        "verified_nodes": verified,
        "all_candidates": [
            {"node": node_key(nodes[i]), "attribution_estimate": float(est[ci]),
             "true_delta_m": float(true_mean[ci])} for ci, i in enumerate(picks)],
    }
    path = args.out or f"results/e2_attribution_{args.prop}.json"
    json.dump(out, open(path, "w"), indent=1)
    # Derive the array path from the output path so l*-sweep runs of the same
    # property do not overwrite each other's raw arrays.
    np.savez(path.replace(".json", "_raw.npz"), mean_stim=mean_stim, mean_ques=mean_ques,
             attr_stim=attr_stim, attr_ques=attr_ques, true_delta=true_delta,
             picks=np.array(picks), nodes=np.array([node_key(n) for n in nodes]),
             m_clean=m_clean, m_corrupt=m_corrupt)
    print(f"\nattribution mass: {json.dumps(breakdown, indent=1)}")
    print(f"attribution-vs-patching r = {corr:.3f} (K3 fires below 0.4)")
    print(f"verified {len(verified)}/{len(picks)} candidates -> {path}")


if __name__ == "__main__":
    main()
