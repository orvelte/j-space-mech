"""E0.0 + E0.1 — plumbing and lens sanity. Must pass before E1 (spec §4, K0).

E0.0  autograd reaches the embedding through the 48 Gated DeltaNet layers.
      Nothing in E2 works without this and it is the most likely silent failure.
E0.1  lens file assertions (in model.load), the walkthrough's boot-currency
      prompt read out mid-stack, and J-lens vs logit-lens top-5 overlap on
      wikitext snippets.
"""

from __future__ import annotations

import argparse
import json

import torch

from jspace import hooks, lens_ops, model as jmodel


def top_k(bundle, scores, k=5):
    return [bundle.tok.decode([t]) for t in scores.topk(k).indices.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="4b", choices=list(jmodel.VARIANTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report: dict = {"variant": args.variant}
    b = jmodel.load(args.variant)
    report["lens"] = {
        "n_prompts": b.lens.n_prompts,
        "d_model": b.lens.d_model,
        "n_source_layers": len(b.lens.source_layers),
        "source_layers": [b.lens.source_layers[0], b.lens.source_layers[-1]],
    }
    report["arch"] = {
        "n_layers": b.n_layers,
        "n_heads": b.n_heads,
        "head_dim": b.head_dim,
        "full_attn_layers": b.full_attn_layers,
        "n_gdn": len(b.gdn_layers),
    }
    print(json.dumps(report, indent=1))

    # ---- E0.0: gradient must flow through the GDN stack -------------------
    ids = b.model.encode("The capital of France is Paris, a city known for")
    probe_layer = b.n_layers // 2
    token_id = b.tok.encode(" Paris")[0]
    with hooks.NodeCache(b, layers=[0, probe_layer], build_graph=True) as cache:
        b.model.forward(ids)
        m = lens_ops.coordinate(b, cache.resid[probe_layer][0, -1], probe_layer, token_id)
        g0 = torch.autograd.grad(m, cache.resid[0], retain_graph=False)[0]
    n_gdn_below = sum(1 for l in range(probe_layer) if not b.is_full_attn(l))
    e0_0 = {
        "metric": float(m),
        "grad_norm_at_layer0": float(g0.norm()),
        "grad_finite": bool(torch.isfinite(g0).all()),
        "gdn_layers_traversed": n_gdn_below,
    }
    e0_0["pass"] = e0_0["grad_finite"] and e0_0["grad_norm_at_layer0"] > 0
    report["E0.0"] = e0_0
    print("E0.0", json.dumps(e0_0))

    # ---- E0.1a: the walkthrough's boot-currency prompt ---------------------
    # Two hops: at the *boot* token the intermediate is Italy; the currency
    # itself only surfaces at the final token, where the answer is being formed.
    prompt = "Fact: The currency used in the country shaped like a boot is"
    layers = list(b.lens.source_layers)
    checks = {"boot_pos": (-2, ("italy", "italia", "\u610f\u5927\u5229")),
              "final_pos": (-1, ("euro", "lira", "lire"))}
    boot = {}
    for label, (pos, targets) in checks.items():
        jl, model_logits, _ = b.lens.apply(b.model, prompt, layers=layers, positions=[pos])
        ll, _, _ = b.lens.apply(
            b.model, prompt, layers=layers, positions=[pos], use_jacobian=False
        )
        jl_top = {l: top_k(b, jl[l][0]) for l in layers}
        ll_top = {l: top_k(b, ll[l][0]) for l in layers}

        def first_hit(d):
            for l in layers:
                if any(any(t in w.lower() for t in targets) for w in d[l]):
                    return l
            return None

        j_hit, l_hit = first_hit(jl_top), first_hit(ll_top)
        boot[label] = {
            "position": pos,
            "targets": list(targets),
            "jlens_first_layer": j_hit,
            "logit_lens_first_layer": l_hit,
            "jlens_precedes_logit_lens": (
                j_hit is not None and (l_hit is None or j_hit < l_hit)
            ),
            "jlens_top5": {l: jl_top[l] for l in layers},
            "logit_lens_top5": {l: ll_top[l] for l in layers},
            "model_top5": top_k(b, model_logits[0]),
        }
    boot["pass"] = any(v["jlens_precedes_logit_lens"] for v in boot.values() if isinstance(v, dict))
    report["E0.1a"] = boot
    print("E0.1a", json.dumps({k: {kk: vv for kk, vv in v.items() if "top5" not in kk}
                               for k, v in boot.items() if isinstance(v, dict)}, indent=1))
    print("E0.1a pass:", boot["pass"])

    # ---- E0.1b: late-layer J-lens vs logit-lens top-5 overlap --------------
    from jlens.examples import load_wikitext_prompts

    snippets = load_wikitext_prompts(10)
    late = b.n_layers - 2
    overlaps = []
    for s in snippets:
        text = " ".join(s.split()[:60])
        jl, _, _ = b.lens.apply(b.model, text, layers=[late], positions=[-2])
        ll, _, _ = b.lens.apply(
            b.model, text, layers=[late], positions=[-2], use_jacobian=False
        )
        a = set(jl[late][0].topk(5).indices.tolist())
        c = set(ll[late][0].topk(5).indices.tolist())
        overlaps.append(len(a & c) / 5)
    mean_overlap = sum(overlaps) / len(overlaps)
    report["E0.1b"] = {
        "layer": late,
        "per_snippet_overlap": overlaps,
        "mean_overlap": mean_overlap,
        "pass_60pct": mean_overlap >= 0.60,
        "K0_fires_below_40pct": mean_overlap < 0.40,
    }
    print("E0.1b", json.dumps(report["E0.1b"], indent=1))

    out = args.out or f"results/e0_lens_sanity_{args.variant}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
